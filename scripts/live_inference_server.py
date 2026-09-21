from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import json
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "fd001_models.joblib"

sys.path.insert(0, str(ROOT))

from cmapss_maintenance.features import add_health_features
from cmapss_maintenance.config import COLUMNS, ExperimentConfig


bundle = joblib.load(MODEL_PATH)

REGRESSION_MODEL = bundle["regression_model"]
MAINTENANCE_MODEL = bundle["maintenance_model"]
FEATURE_COLUMNS = bundle["feature_columns"]
THRESHOLD = float(bundle["maintenance_threshold"])

WINDOW = ExperimentConfig().feature_window


def parse_csv(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    if not lines:
        raise ValueError("CSV input is empty.")

    # Named-header CSV
    if lines[0].lower().startswith("unit_number"):
        from io import StringIO
        df = pd.read_csv(StringIO(text))
    else:
        # NASA C-MAPSS raw format
        from io import StringIO
        df = pd.read_csv(
            StringIO(text),
            sep=r"\s+|,",
            engine="python",
            header=None,
        )

        if df.shape[1] != len(COLUMNS):
            raise ValueError(
                f"Expected {len(COLUMNS)} columns, but received {df.shape[1]}."
            )

        df.columns = COLUMNS

    missing = [c for c in COLUMNS if c not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df[COLUMNS].copy()


def classify_risk(probability):
    if probability >= 0.60:
        return "critical"
    if probability >= THRESHOLD:
        return "watch"
    return "stable"


def predict(df, explain=False):
    if len(df) < WINDOW:
        raise ValueError(
            f"At least {WINDOW} sequential cycles are required "
            f"for the trained {WINDOW}-cycle model."
        )

    df = df.sort_values(["unit_number", "time_in_cycles"]).copy()

    featured = add_health_features(df, FEATURE_COLUMNS, window=WINDOW)

    X = featured[FEATURE_COLUMNS]

    rul_prediction = REGRESSION_MODEL.predict(X)

    if hasattr(MAINTENANCE_MODEL, "predict_proba"):
        maintenance_probability = MAINTENANCE_MODEL.predict_proba(X)[:, 1]
    else:
        maintenance_probability = MAINTENANCE_MODEL.predict(X)

    results = []

    for i in range(len(featured)):
        probability = float(maintenance_probability[i])

        results.append({
            "unitNumber": int(featured.iloc[i]["unit_number"]),
            "cycle": int(featured.iloc[i]["time_in_cycles"]),
            "predictedRUL": round(max(0.0, float(rul_prediction[i])), 2),
            "maintenanceProbability": round(probability, 4),
            "status": classify_risk(probability),
        })

    latest = results[-1]

    response = {
        "success": True,
        "model": "HistGradientBoostingRegressor",
        "maintenanceModel": "RandomForestClassifier",
        "featureWindow": WINDOW,
        "threshold": THRESHOLD,
        "rowsProcessed": len(df),
        "latest": latest,
        "predictions": results,
    }

    if explain:
        try:
            import shap

            explainer = shap.Explainer(REGRESSION_MODEL)
            sample = X.iloc[[-1]]
            shap_values = explainer(sample)

            values = shap_values.values[0]

            explanation = sorted(
                zip(FEATURE_COLUMNS, values),
                key=lambda item: abs(float(item[1])),
                reverse=True,
            )[:5]

            response["explanation"] = [
                {
                    "feature": feature,
                    "impact": round(float(value), 4),
                }
                for feature, value in explanation
            ]

        except Exception as exc:
            response["explanationError"] = str(exc)

    return response


class Handler(BaseHTTPRequestHandler):

    def _send(self, status, payload):
        body = json.dumps(payload).encode("utf-8")

        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send(200, {"success": True})

    def do_GET(self):
        if self.path == "/health":
            self._send(
                200,
                {
                    "success": True,
                    "service": "EngineGuard Live Inference",
                    "modelLoaded": True,
                    "featureWindow": WINDOW,
                    "threshold": THRESHOLD,
                },
            )
            return

        self._send(404, {"success": False, "error": "Not found"})

    def do_POST(self):
        if self.path != "/predict":
            self._send(404, {"success": False, "error": "Not found"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)

            payload = json.loads(raw.decode("utf-8"))

            csv_text = payload.get("csv", "")
            explain = bool(payload.get("explain", False))

            if not csv_text:
                raise ValueError("CSV data is required.")

            df = parse_csv(csv_text)

            result = predict(df, explain=explain)

            self._send(200, result)

        except Exception as exc:
            self._send(
                400,
                {
                    "success": False,
                    "error": str(exc),
                },
            )


if __name__ == "__main__":
    print("=" * 60)
    print("EngineGuard Live Inference Server")
    print("=" * 60)
    print(f"Model: {type(REGRESSION_MODEL).__name__}")
    print(f"Features: {len(FEATURE_COLUMNS)}")
    print(f"Window: {WINDOW} cycles")
    print(f"Threshold: {THRESHOLD}")
    print("Server: http://127.0.0.1:8000")
    print("=" * 60)

    server = HTTPServer(("127.0.0.1", 8000), Handler)
    server.serve_forever()
