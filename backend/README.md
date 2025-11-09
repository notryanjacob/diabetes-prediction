# Diabetes Prediction API

This FastAPI app loads the trained scikit-learn model (stored as a `.pkl` file) and exposes the `/predict` endpoint that the Life Check AI frontend calls.

## Getting started

1. Copy your trained model into `backend/models/diabetes_prediction_model.pkl`.
2. Create a virtual environment (optional but recommended) and install the dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r backend/requirements.txt
   ```
3. Start the API locally:
   ```bash
   uvicorn backend.main:app --reload --port 8000
   ```

The service now listens on `http://localhost:8000`. Update the `.env` file at the project root if you need to:

- Point the frontend to a different API base URL (`VITE_API_BASE_URL`).
- Allow additional origins to call the backend (`ALLOWED_ORIGINS=http://localhost:5173,...`). Set `*` to allow any origin (cookies are disabled automatically in that case).
- Enable Gemini-powered workout/diet plans by setting `GEMINI_API_KEY` (and optionally `GEMINI_MODEL`, `PLAN_THRESHOLD`).

## Endpoints

- `GET /health`: lightweight readiness probe. Returns whether the model file was loaded.
- `GET /predict/sample-features`: returns a ready-to-use payload that satisfies the schema (useful when testing with tools like cURL or Postman).
- `POST /predict/sample`: runs the bundled sample payload through the model so you can verify the backend without filling out the questionnaire.
- `POST /plan`: accepts `{ features, probability, planType }`, calls Gemini via LangChain, and returns personalized workout and/or diet plans (requires `GEMINI_API_KEY`).
- `POST /plan/sample`: quickly generate a plan using the bundled sample features; accepts optional `probability` and `plan_type` query params.
- `POST /predict`: accepts the full set of features collected by the chatbot and returns the probability, risk level, predicted label (based on the stored threshold), and a diabetes score from 0–10.
