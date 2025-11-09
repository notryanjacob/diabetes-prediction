from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, conint, confloat

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:  # pragma: no cover - optional dependency during dev
    ChatGoogleGenerativeAI = None  # type: ignore

logger = logging.getLogger("diabetes-api")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Diabetes Prediction API",
    version="0.1.0",
    description="Serves diabetes risk predictions for the Life Check AI frontend.",
)

RAW_ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "*",
)
ALLOWED_ORIGINS = [origin.strip() for origin in RAW_ALLOWED_ORIGINS.split(",") if origin.strip()]
ALLOW_ALL_ORIGINS = not ALLOWED_ORIGINS or ALLOWED_ORIGINS == ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if ALLOW_ALL_ORIGINS else ALLOWED_ORIGINS,
    allow_credentials=not ALLOW_ALL_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path("/Users/ryanjacob/Downloads/diabetes_prediction_model(2).pkl")
BEST_THRESHOLD = 0.674
PLAN_THRESHOLD = float(os.getenv("PLAN_THRESHOLD", "0.4"))
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

SELECTED_FEATURES = [
    "HighBP",
    "HighChol",
    "BMI",
    "Smoker",
    "Stroke",
    "HeartDiseaseorAttack",
    "CholCheck",
    "PhysActivity",
    "Fruits",
    "Veggies",
    "HvyAlcoholConsump",
    "GenHlth",
    "Age",
    "DiffWalk",
    "PhysHlth",
    "MentHlth",
    "NoDocbcCost",
    "Education",
    "Income",
    "Sex",
    "AnyHealthcare",
]

BOOLEAN_COLUMNS = [
    "HighBP",
    "HighChol",
    "Smoker",
    "Stroke",
    "HeartDiseaseorAttack",
    "CholCheck",
    "PhysActivity",
    "Fruits",
    "Veggies",
    "HvyAlcoholConsump",
    "DiffWalk",
    "NoDocbcCost",
    "AnyHealthcare",
    "Sex",
]

AGE_BUCKETS = [
    (18, 24, 21),
    (25, 29, 27),
    (30, 34, 32),
    (35, 39, 37),
    (40, 44, 42),
    (45, 49, 47),
    (50, 54, 52),
    (55, 59, 57),
    (60, 64, 62),
    (65, 69, 67),
    (70, 74, 72),
    (75, 79, 77),
]

model = None
plan_llm: ChatGoogleGenerativeAI | None = None

PlanType = Literal["workout", "diet", "both"]


class DiabetesFeatures(BaseModel):
    HighBP: bool = Field(..., description="Diagnosed with high blood pressure")
    HighChol: bool = Field(..., description="Diagnosed with high cholesterol")
    BMI: confloat(ge=10, le=60) = Field(..., description="Body Mass Index")
    Smoker: bool = Field(..., description="Has smoked at least 100 cigarettes in life")
    Stroke: bool = Field(..., description="Has ever been told of having a stroke")
    HeartDiseaseorAttack: bool = Field(..., description="Has coronary heart disease or myocardial infarction history")
    CholCheck: bool = Field(..., description="Had cholesterol check within last 5 years")
    PhysActivity: bool = Field(..., description="Had physical activity in past 30 days (excluding job)")
    Fruits: bool = Field(..., description="Consumes fruits at least once per day")
    Veggies: bool = Field(..., description="Consumes vegetables at least once per day")
    HvyAlcoholConsump: bool = Field(..., description="Heavy alcohol consumption (men >14 drinks/week, women >7)")
    GenHlth: conint(ge=1, le=5) = Field(..., description="General health rating (1=excellent ... 5=poor)")
    Age: conint(ge=18, le=120) = Field(..., description="Age in years")
    DiffWalk: bool = Field(..., description="Serious difficulty walking or climbing stairs")
    PhysHlth: conint(ge=0, le=30) = Field(..., description="Number of days with poor physical health (0-30)")
    MentHlth: conint(ge=0, le=30) = Field(..., description="Number of days with poor mental health (0-30)")
    NoDocbcCost: bool = Field(..., description="Unable to see doctor because of cost in past 12 months")
    Education: conint(ge=1, le=6) = Field(..., description="Education level (1=never attended ... 6=college graduate)")
    Income: conint(ge=1, le=8) = Field(..., description="Income level (1=<$10k ... 8=>$75k)")
    Sex: bool = Field(..., description="True if male, False if female")
    AnyHealthcare: bool = Field(..., description="Has any kind of health coverage")


SAMPLE_FEATURES = DiabetesFeatures(
    HighBP=True,
    HighChol=False,
    BMI=29.4,
    Smoker=True,
    Stroke=False,
    HeartDiseaseorAttack=False,
    CholCheck=True,
    PhysActivity=True,
    Fruits=True,
    Veggies=True,
    HvyAlcoholConsump=False,
    GenHlth=2,
    Age=42,
    DiffWalk=False,
    PhysHlth=2,
    MentHlth=4,
    NoDocbcCost=False,
    Education=5,
    Income=6,
    Sex=True,
    AnyHealthcare=True,
)


class HealthPlan(BaseModel):
    planType: PlanType
    workoutPlan: str | None = None
    dietPlan: str | None = None
    prompt: str


class PlanRequest(BaseModel):
    features: DiabetesFeatures
    probability: float = Field(..., ge=0.0, le=1.0)
    planType: PlanType = "both"


class PredictionResponse(BaseModel):
    probability: float
    riskLevel: Literal["low", "moderate", "high"]
    diabetesScore: float
    predictedLabel: Literal[0, 1]
    message: str | None = None
    plan: HealthPlan | None = None


def load_model() -> object:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}. "
            "Add your 'diabetes_prediction_model.pkl' file to continue."
        )
    logger.info("Loading model from %s", MODEL_PATH)
    return joblib.load(MODEL_PATH)


def map_age_to_midpoint(age: int) -> int:
    for lower, upper, midpoint in AGE_BUCKETS:
        if lower <= age <= upper:
            return midpoint
    # Age 80+ falls outside predefined buckets; align with dataset midpoint
    return 82


def preprocess_input(features: DiabetesFeatures) -> pd.DataFrame:
    payload = features.model_dump()

    for column in BOOLEAN_COLUMNS:
        payload[column] = int(payload[column])

    payload["Age"] = map_age_to_midpoint(payload["Age"])

    df = pd.DataFrame([payload])
    df = df[SELECTED_FEATURES]
    return df


def get_risk_level(probability: float) -> Literal["low", "moderate", "high"]:
    if probability < 0.3:
        return "low"
    if probability < 0.6:
        return "moderate"
    return "high"


def build_message(risk_level: Literal["low", "moderate", "high"]) -> str:
    match risk_level:
        case "low":
            return "Your responses indicate a low diabetes risk profile. Maintain your healthy habits."
        case "moderate":
            return "Your risk is moderate. Consider following up with routine screening and lifestyle adjustments."
        case "high":
            return "Your results suggest an elevated risk. Please consult a healthcare professional for diagnostic testing."


def initialize_plan_llm() -> None:
    global plan_llm
    if GEMINI_API_KEY is None:
        logger.info("GEMINI_API_KEY not configured; skipping health plan generation setup.")
        plan_llm = None
        return
    if ChatGoogleGenerativeAI is None:
        logger.warning("langchain-google-genai not installed; health plans unavailable.")
        plan_llm = None
        return

    os.environ.setdefault("GOOGLE_API_KEY", GEMINI_API_KEY)
    try:
        plan_llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL_NAME,
            temperature=0.4,
            max_output_tokens=1024,
        )
        logger.info("Gemini model '%s' initialized for plan generation.", GEMINI_MODEL_NAME)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to initialize Gemini model: %s", exc)
        plan_llm = None


def build_plan_prompt(features: DiabetesFeatures, probability: float, plan_type: PlanType) -> str:
    feature_summary = json.dumps(features.model_dump(), indent=2)
    percent = probability * 100
    plan_focus = {
        "workout": "a practical 7-day workout routine (include rest guidance, intensity, safety tips)",
        "diet": "a 7-day nutrition plan (breakfast/lunch/dinner/snacks, hydration, portion guidance)",
        "both": "a practical 7-day workout routine and a matching nutrition plan",
    }[plan_type]

    schema_lines = []
    if plan_type in ("workout", "both"):
        schema_lines.append(
            '  "workoutPlan": string  // markdown with day-by-day workouts, rest, intensity, safety tips'
        )
    if plan_type in ("diet", "both"):
        schema_lines.append(
            '  "dietPlan": string     // markdown with meals, hydration targets, portion reminders'
        )

    schema = "{\n" + ",\n".join(schema_lines) + "\n}"

    return (
        "You are an empathetic AI health coach. A patient completed a diabetes assessment "
        f"and has an estimated risk of {percent:.1f}%. Using the structured data below, craft "
        f"{plan_focus}. Keep the tone encouraging, actionable, and suitable for adults."
        "\n\nRespond ONLY with JSON matching this schema:\n"
        f"{schema}\n"
        "Avoid clinical diagnoses—focus on lifestyle guidance that can complement professional care."
        "\n\nPatient profile:\n"
        f"```json\n{feature_summary}\n```"
    )


def extract_json_block(raw_text: str) -> str:
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return raw_text
    return raw_text[start : end + 1]


def generate_health_plan(
    features: DiabetesFeatures, probability: float, plan_type: PlanType = "both"
) -> HealthPlan:
    if plan_llm is None:
        raise RuntimeError("Gemini health plan generator is not configured.")

    prompt = build_plan_prompt(features, probability, plan_type)
    response = plan_llm.invoke(prompt)
    print(response)
    content = getattr(response, "content", response)
    if isinstance(content, list):
        raw_text = "".join(
            chunk.get("text", "") if isinstance(chunk, dict) else str(chunk) for chunk in content
        )
    else:
        raw_text = str(content)
    logger.info("Gemini raw response: %s", raw_text)

    candidate = extract_json_block(raw_text)
    workout_text: str | None = None
    diet_text: str | None = None
    try:
        parsed = json.loads(candidate)
        if plan_type in ("workout", "both"):
            workout_text = str(parsed.get("workoutPlan") or "").strip() or None
        if plan_type in ("diet", "both"):
            diet_text = str(parsed.get("dietPlan") or "").strip() or None
    except json.JSONDecodeError:
        logger.warning("Failed to parse Gemini response as JSON. Returning raw text.")
        if plan_type in ("workout", "both"):
            workout_text = raw_text.strip()
        if plan_type in ("diet", "both"):
            diet_text = raw_text.strip()
    else:
        fallback_text = raw_text.strip() if raw_text else None
        if plan_type in ("workout", "both") and not workout_text and fallback_text:
            workout_text = fallback_text
        if plan_type in ("diet", "both") and not diet_text and fallback_text:
            diet_text = fallback_text

    return HealthPlan(
        planType=plan_type,
        workoutPlan=workout_text,
        dietPlan=diet_text,
        prompt=prompt,
    )


@app.on_event("startup")
def initialize_model() -> None:
    global model
    try:
        model = load_model()
    except FileNotFoundError as exc:
        logger.warning(str(exc))
        model = None
    initialize_plan_llm()


@app.get("/health")
def health_check() -> dict[str, str | bool]:
    return {"status": "ok", "modelLoaded": model is not None}


def execute_prediction(features: DiabetesFeatures) -> PredictionResponse:
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Ensure the pickle file exists in backend/models.",
        )

    try:
        feature_df = preprocess_input(features)
        probabilities = model.predict_proba(feature_df)[:, 1]
        probability = float(np.clip(probabilities[0], 0.0, 1.0))
    except Exception as exc:  # noqa: BLE001
        logger.exception("Prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail="Prediction failed") from exc

    predicted_label = 1 if probability >= BEST_THRESHOLD else 0
    risk_level = get_risk_level(probability)
    diabetes_score = round(probability * 10, 1)
    return PredictionResponse(
        probability=probability,
        riskLevel=risk_level,
        diabetesScore=diabetes_score,
        predictedLabel=predicted_label,
        message=build_message(risk_level),
        plan=None,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict_diabetes(features: DiabetesFeatures) -> PredictionResponse:
    return execute_prediction(features)


@app.get("/predict/sample-features", response_model=DiabetesFeatures)
def get_sample_features() -> DiabetesFeatures:
    return SAMPLE_FEATURES


@app.post("/plan", response_model=HealthPlan)
def create_plan(plan_request: PlanRequest) -> HealthPlan:
    if plan_request.probability < PLAN_THRESHOLD:
        raise HTTPException(
            status_code=400,
            detail=f"Plan generation is only available for probabilities >= {PLAN_THRESHOLD:.2f}.",
        )

    try:
        return generate_health_plan(
            plan_request.features, plan_request.probability, plan_request.planType
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Health plan generation failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to generate plan") from exc


@app.post("/plan/sample", response_model=HealthPlan)
def sample_plan(
    probability: float | None = None,
    plan_type: PlanType = "both",
) -> HealthPlan:
    effective_probability = probability if probability is not None else max(PLAN_THRESHOLD, 0.5)
    if effective_probability < PLAN_THRESHOLD:
        raise HTTPException(
            status_code=400,
            detail=f"Probability must be >= {PLAN_THRESHOLD:.2f} to request a plan.",
        )

    try:
        return generate_health_plan(SAMPLE_FEATURES, effective_probability, plan_type)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Sample plan generation failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to generate plan") from exc


@app.post("/predict/sample", response_model=PredictionResponse)
def predict_sample() -> PredictionResponse:
    return execute_prediction(SAMPLE_FEATURES)
