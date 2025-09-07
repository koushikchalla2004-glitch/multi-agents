from fastapi import FastAPI, UploadFile, Form
from pydantic import BaseModel
from pathlib import Path
import pandas as pd, numpy as np, json, joblib, os
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, f1_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from typing import Optional

# Simple artifact root (mount a persistent volume in production)
ART_DIR = Path(os.getenv("ARTIFACT_ROOT", "./artifacts"))
ART_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="One-Click Data Team – Worker API")

class IngestIn(BaseModel):
    project_id: str
    storage_path: str  # local path or mounted path to uploaded file

class EdaIn(BaseModel):
    project_id: str
    target: Optional[str] = None

class TrainIn(BaseModel):
    project_id: str
    task_type: str
    target: str

class ReportIn(BaseModel):
    project_id: str

def proj_dir(pid: str) -> Path:
    d = ART_DIR / pid
    d.mkdir(parents=True, exist_ok=True)
    return d

def load_table(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf == ".csv":
        return pd.read_csv(path)
    if suf in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if suf == ".parquet":
        return pd.read_parquet(path)
    if suf == ".json":
        return pd.read_json(path)
    raise ValueError(f"Unsupported format: {suf}")

@app.post("/ingest")
def ingest(p: IngestIn):
    src = Path(p.storage_path.replace("storage://", ""))
    df = load_table(src)
    info = {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "columns": df.columns.tolist(),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()}
    }
    df = df.dropna(axis=1, how="all")
    cleaned = proj_dir(p.project_id) / "cleaned.csv"
    df.to_csv(cleaned, index=False)
    (proj_dir(p.project_id) / "schema.json").write_text(json.dumps(info, indent=2))
    return {"status": "ok", "cleaned_path": str(cleaned), "schema": info}

@app.post("/eda")
def eda(p: EdaIn):
    import matplotlib.pyplot as plt
    df = pd.read_csv(proj_dir(p.project_id) / "cleaned.csv")
    notes = [f"Rows={len(df)} Cols={df.shape[1]}"]
    notes.append(f"Nulls: {df.isna().sum().to_dict()}")
    num = df.select_dtypes(include=np.number)
    if not num.empty:
        corr = num.corr(numeric_only=True)
        fig = plt.figure()
        plt.imshow(corr, interpolation='nearest')
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=6)
        plt.yticks(range(len(corr.columns)), corr.columns, fontsize=6)
        plt.colorbar()
        plt.tight_layout()
        png = proj_dir(p.project_id) / "corr.png"
        fig.savefig(png, dpi=160)
        plt.close(fig)
        notes.append("Saved corr.png")
    if p.target and p.target in df.columns:
        notes.append(f"Target={p.target} unique={df[p.target].nunique()}")
    (proj_dir(p.project_id) / "eda.txt").write_text("\n".join(notes))
    return {"status": "ok", "eda_notes": notes}

@app.post("/train")
def train(p: TrainIn):
    df = pd.read_csv(proj_dir(p.project_id) / "cleaned.csv")
    assert p.target in df.columns, f"Target {p.target} not in columns"
    X = pd.get_dummies(df.drop(columns=[p.target]), drop_first=True)
    y = df[p.target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    metrics = {}
    if p.task_type == "classification" or (y.dtype == object or y.nunique() < 20):
        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        metrics = {"accuracy": float(accuracy_score(y_test, pred)),
                   "f1_macro": float(f1_score(y_test, pred, average='macro'))}
    else:
        model = RandomForestRegressor(n_estimators=300, random_state=42)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        metrics = {"r2": float(r2_score(y_test, pred)),
                   "rmse": float(mean_squared_error(y_test, pred, squared=False))}

    model_path = proj_dir(p.project_id) / "model.pkl"
    joblib.dump(model, model_path)
    (proj_dir(p.project_id) / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return {"status": "ok", "model_path": str(model_path), "metrics": metrics}

@app.post("/report")
def report(p: ReportIn):
    proj = proj_dir(p.project_id)
    schema = json.loads((proj / "schema.json").read_text())
    metrics = json.loads((proj / "metrics.json").read_text()) if (proj / "metrics.json").exists() else {}
    has_corr = (proj / "corr.png").exists()
    html = f"""<html><head><meta charset='utf-8'><title>{p.project_id} Report</title></head>
<body>
<h1>Automated Report – {p.project_id}</h1>
<h2>Dataset</h2><pre>{json.dumps(schema, indent=2)}</pre>
<h2>Metrics</h2><pre>{json.dumps(metrics, indent=2)}</pre>
{("<h2>Correlation</h2><img src='corr.png' width='800'/>" if has_corr else "")}
</body></html>"""
    (proj / "report.html").write_text(html)
    return {"status": "ok", "links": {"report_html": f"/static/{p.project_id}/report.html"}}
