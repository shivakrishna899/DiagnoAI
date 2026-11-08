
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from diagnoai.inference import (
    ModelNotReadyError,
    Predictor,
    image_from_bytes,
)


# --------------------------------------------------
# Application configuration
# --------------------------------------------------

app = FastAPI(
    title="DiagnoAI Research API",
    version="0.2.0",
    description=(
        "Educational multimodal prediction prototype. "
        "Not a clinical diagnostic service."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# --------------------------------------------------
# Model initialization
# --------------------------------------------------

predictor = Predictor(
    Path("models/diagnoai.pt")
)

MAX_UPLOAD_BYTES = 10 * 1024 * 1024


# --------------------------------------------------
# Health endpoint
# --------------------------------------------------

@app.get("/health")
def health() -> dict[str, Any]:

    return {
        "status": "ready" if predictor.ready else "degraded",
        "model_loaded": predictor.ready,
        "load_error": predictor.load_error,
        "disclaimer": (
            "Research prototype only. "
            "Not a medical diagnosis."
        ),
    }


# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    symptoms: str = Form(
        "{}",
        description=(
            "JSON object containing symptom states. "
            "Allowed states: yes, no, unknown."
        ),
    ),
) -> dict[str, Any]:

    # Check model availability first.
    if not predictor.ready:
        raise HTTPException(
            status_code=503,
            detail=(
                predictor.load_error
                or "Trained model is not available."
            ),
        )

    try:
        # ------------------------------------------
        # 1. Read uploaded image safely
        # ------------------------------------------

        data = await image.read(
            MAX_UPLOAD_BYTES + 1
        )

        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail="Maximum upload size is 10 MB.",
            )

        if not data:
            raise HTTPException(
                status_code=400,
                detail="Uploaded image is empty.",
            )

        # ------------------------------------------
        # 2. Parse symptom states
        # ------------------------------------------

        try:
            symptom_states = json.loads(symptoms)

        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=400,
                detail="Symptoms must contain valid JSON.",
            ) from exc

        if not isinstance(symptom_states, dict):
            raise HTTPException(
                status_code=400,
                detail=(
                    "Symptoms must be a JSON object "
                    "mapping symptom names to states."
                ),
            )

        # ------------------------------------------
        # 3. Decode and validate image
        # ------------------------------------------

        try:
            decoded_image = image_from_bytes(data)

        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

        # ------------------------------------------
        # 4. Run multimodal inference
        # ------------------------------------------

        result = predictor.predict(
            image=decoded_image,
            symptom_states=symptom_states,
            top_k=3,
            include_explanations=True,
        )

        return result

    except HTTPException:
        raise

    except ModelNotReadyError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        # Keep internal exception details out of API responses.
        print(f"Prediction error: {exc}")

        raise HTTPException(
            status_code=500,
            detail=(
                "Prediction failed. "
                "Check local server logs for details."
            ),
        ) from exc

    finally:
        await image.close()
