"""
SpamShield API — FastAPI backend.

Run from the project root with:
    uvicorn app.main:app --reload --port 8000

Docs available at http://localhost:8000/docs
"""

import csv
import io
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

# Allow `import src.predict` when running from project root
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.predict import predict_email, predict_batch, available_models, model_metrics

app = FastAPI(title="SpamShield API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this before deploying publicly
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# In-memory stats (resets on server restart; swap for a DB if you need it
# to persist)
# ---------------------------------------------------------------------------
STATS = {"total": 0, "spam": 0, "not_spam": 0}


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class EmailRequest(BaseModel):
    subject: str = ""
    body: str
    model: Optional[str] = None  # "Naive Bayes" | "Logistic Regression" | "Linear SVM"


class EmailResponse(BaseModel):
    label: str
    confidence: float
    model_used: str
    top_indicators: list
    cleaned_text_preview: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/models")
def list_models():
    """Available models and their held-out test metrics from training."""
    return {"available_models": available_models(), "metrics": model_metrics()}


@app.post("/predict", response_model=EmailResponse)
def predict(req: EmailRequest):
    if not req.body.strip() and not req.subject.strip():
        raise HTTPException(status_code=400, detail="Provide a subject and/or body.")

    result = predict_email(req.subject, req.body, req.model)

    STATS["total"] += 1
    if result["label"] == "spam":
        STATS["spam"] += 1
    else:
        STATS["not_spam"] += 1

    return result


@app.post("/predict/batch")
async def predict_batch_csv(file: UploadFile = File(...), model: Optional[str] = None):
    """Upload a CSV with a 'text' column (optionally 'subject'). Returns
    predictions for every row and a downloadable CSV of results."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file.")

    raw = await file.read()
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8", errors="ignore")))
    rows = list(reader)

    if not rows:
        raise HTTPException(status_code=400, detail="CSV is empty.")
    if "text" not in reader.fieldnames:
        raise HTTPException(
            status_code=400,
            detail=f"CSV must have a 'text' column. Found columns: {reader.fieldnames}",
        )

    texts = [
        f"{row.get('subject', '')} {row.get('subject', '')} {row['text']}"
        for row in rows
    ]
    predictions = predict_batch(texts, model)

    for row, pred in zip(rows, predictions):
        row["predicted_label"] = pred["label"]
        row["confidence"] = pred["confidence"]
        STATS["total"] += 1
        STATS["spam" if pred["label"] == "spam" else "not_spam"] += 1

    spam_count = sum(1 for p in predictions if p["label"] == "spam")

    output = io.StringIO()
    fieldnames = list(rows[0].keys())
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    output.seek(0)

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=predictions.csv",
            "X-Total-Rows": str(len(rows)),
            "X-Spam-Count": str(spam_count),
        },
    )


@app.get("/stats")
def stats():
    total = STATS["total"] or 1  # avoid div by zero
    return {
        **STATS,
        "spam_percentage": round(100 * STATS["spam"] / total, 2) if STATS["total"] else 0,
    }


# ---------------------------------------------------------------------------
# Serve the simple frontend (index.html + static assets) from /
# ---------------------------------------------------------------------------
STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def serve_index():
    return FileResponse(STATIC_DIR / "index.html")
