from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
from typing import Optional
import os, json, tempfile, requests

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, f1_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# ---------- Config ----------
# Writable artifacts directory (set in Render/Railway env as ARTIFACT_ROOT=./artifacts)
ART_DIR = Path(os.getenv("ARTIFACT_ROOT", "./artifacts"))
ART_DIR.mkdir(parents=True, exist_ok=True)

# Optional: Supabase env to download files that were uploaded to Supabase Storage
SUPABASE_URL = os.getenv("SUPABASE_URL", "")              # e.g., https://xxxx.supabase.co
SUPABASE_SERVICE_ROLE = os.getenv("SUPABASE_SERVICE_ROLE", "")  # service_role key (server-side only)

app = FastAPI(title="One-Click Data Team – Small Worker")

# Serve artifacts (HTML report, images) at /static/<project_id>/...
from fastapi.staticfiles import StaticFiles
app.mount("/static", StaticFiles(directory=ART_DIR, html=True), name="static")

# ---------- Models ----------
class IngestIn(BaseModel):
    project_id: str
    storage_path: str  # e.g., storage://uploads/<object_name> (Supabase) or local path

class EdaIn(BaseModel):
    project_id: str
    target: Optional[str] = None

class TrainIn(BaseModel):
    project_id: str
    task_type: str           # classification | regression (heuristic if not sure)
    target: str

class ReportIn(BaseModel):
    project_id: str

# ---------- Helpers ----------
def proj_dir(pid: str) -> Path:
    d = ART_DIR / pid
    d.mkdir(parents=True, exist_ok=True)
    return d

def load_table(local_path: Path) -> pd.DataFrame:
    suf = local_path.suffix.lower()
    if suf == ".csv":
        return pd.read_csv(local_path)
    if suf in (".xlsx", ".xls"):
        return pd.read_excel(local_path)
    if suf == ".parquet":
        return pd.read_parquet(local_path)
    if suf == ".json":
        return pd.read_json(local_path)
    raise ValueError(f"Unsupported format: {suf}")

def _download_from_supabase(src_marker: str) -> Path:
    """Download an object from Supabase Storage 'uploads' bucket to a temp file."""
    assert SUPABASE_URL and SUPABASE_SERVICE_ROLE, "Set SUPABASE_URL and SUPABASE_SERVICE_ROLE env vars."
    obj = src_marker[len("storage://uploads/"):]  # strip marker
    url = f"{SUPABASE_URL}/storage/v1/object/{'uploads'}/{obj}"
    headers = {"Authorization": f"Bearer {SUPABASE_SERVICE_ROLE}", "apikey": SUPABASE_SERVICE_ROLE}
    with requests.get(url, headers=headers, stream=True, timeout=300) as r:
        r.raise_for_status()
        suffix = Path(obj).suffix or ".bin"
        fd, tmp_path = tempfile.mkstemp(prefix="dataset_", suffix=suffix)
        with os.fdopen(fd, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024*1024):
                if chunk:
                    f.write(chunk)
    return Path(tmp_path)

# ---------- Endpoints ----------
@app.post("/ingest")
def ingest(p: IngestIn):
    # Resolve path: either Supabase storage marker or local path
    if p.storage_path.startswith("storage://uploads/"):
        local_src = _download_from_supabase(p.storage_path)
    else:
        local_src = Path(p.storage_path.replace("storage://", ""))

    df = load_table(local_src)

    info = {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "columns": df.columns.tolist(),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()}
    }

    # Simple clean: drop all-NaN columns
    df = df.dropna(axis=1, how="all")

    cleaned = proj_dir(p.project_id) / "cleaned.csv"
    df.to_csv(cleaned, index=False)
    (proj_dir(p.project_id) / "schema.json").write_text(json.dumps(info, indent=2))

    return {"status": "ok", "cleaned_path": str(cleaned), "schema": info}

@app.post("/eda")
def eda(p: EdaIn):
    import matplotlib
    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt

    proj = proj_dir(p.project_id)
    df = pd.read_csv(proj / "cleaned.csv")

    notes = {
        "rows": int(len(df)),
        "cols": int(df.shape[1]),
        "nulls": {k:int(v) for k,v in df.isna().sum().to_dict().items()}
    }

    num = df.select_dtypes(include=np.number)
    if not num.empty and num.shape[1] >= 2:
        corr = num.corr(numeric_only=True)
        fig = plt.figure(figsize=(8,6))
        plt.imshow(corr, interpolation='nearest')
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=6)
        plt.yticks(range(len(corr.columns)), corr.columns, fontsize=6)
        plt.colorbar()
        plt.tight_layout()
        png = proj / "corr.png"
        fig.savefig(png, dpi=150)
        plt.close(fig)
        notes["corr_png"] = str(png)

    if p.target and p.target in df.columns:
        notes["target"] = {"name": p.target, "unique": int(df[p.target].nunique())}

    (proj / "eda.json").write_text(json.dumps(notes, indent=2))
    return {"status": "ok", "eda": notes}

@app.post("/train")
def train(p: TrainIn):
    proj = proj_dir(p.project_id)
    df = pd.read_csv(proj / "cleaned.csv")
    assert p.target in df.columns, f"Target {p.target} not in columns"

    y = df[p.target]
    X = pd.get_dummies(df.drop(columns=[p.target]), drop_first=True)

    # Coerce any remaining object columns to numeric where possible
    for c in list(X.columns):
        if X[c].dtype == "object":
            X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Heuristic: classification if few unique classes or object dtype
    task_type = p.task_type
    if task_type not in ("classification", "regression"):
        task_type = "classification" if (y.dtype == object or y.nunique() <= 20) else "regression"

    metrics = {}
    if task_type == "classification":
        model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1_macro": float(f1_score(y_test, y_pred, average="macro"))
        }
    else:
        model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = {
            "r2": float(r2_score(y_test, y_pred)),
            "rmse": float(mean_squared_error(y_test, y_pred, squared=False))
        }

    model_path = proj / "model.pkl"
    joblib.dump(model, model_path)
    (proj / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return {"status": "ok", "model_path": str(model_path), "metrics": metrics}

@app.post("/report")
def report(p: ReportIn):
    proj = proj_dir(p.project_id)
    schema = json.loads((proj / "schema.json").read_text()) if (proj / "schema.json").exists() else {}
    metrics = json.loads((proj / "metrics.json").read_text()) if (proj / "metrics.json").exists() else {}
    has_corr = (proj / "corr.png").exists()

    html = f"""<html><head><meta charset='utf-8'><title>{p.project_id} Report</title></head>
<body style='font-family:Arial, sans-serif; margin:24px'>
<h1>Automated Report – {p.project_id}</h1>
<h2>Dataset</h2><pre>{json.dumps(schema, indent=2)}</pre>
<h2>Metrics</h2><pre>{json.dumps(metrics, indent=2)}</pre>
{('<h2>Correlation</h2><img src="corr.png" width="800"/>' if has_corr else '')}
<p style='color:#666'>Generated by Small Worker</p>
</body></html>"""
    (proj / "report.html").write_text(html)
    # Served via /static
    return {"status": "ok", "links": {"report_html": f"/static/{p.project_id}/report.html"}}
