# Welcome to your Lovable project

## Project info

**URL**: https://lovable.dev/projects/d6b8cd81-3ed5-4585-bc99-c145dcb70ead

## How can I edit this code?

There are several ways of editing your application.

**Use Lovable**

Simply visit the [Lovable Project](https://lovable.dev/projects/d6b8cd81-3ed5-4585-bc99-c145dcb70ead) and start prompting.

Changes made via Lovable will be committed automatically to this repo.

**Use your preferred IDE**

If you want to work locally using your own IDE, you can clone this repo and push changes. Pushed changes will also be reflected in Lovable.

The only requirement is having Node.js & npm installed - [install with nvm](https://github.com/nvm-sh/nvm#installing-and-updating)

Follow these steps:

```sh
# Step 1: Clone the repository using the project's Git URL.
git clone <YOUR_GIT_URL>

# Step 2: Navigate to the project directory.
cd <YOUR_PROJECT_NAME>

# Step 3: Install the necessary dependencies.
npm i

# Step 4: Start the development server with auto-reloading and an instant preview.
npm run dev
```

**Edit a file directly in GitHub**

- Navigate to the desired file(s).
- Click the "Edit" button (pencil icon) at the top right of the file view.
- Make your changes and commit the changes.

**Use GitHub Codespaces**

- Navigate to the main page of your repository.
- Click on the "Code" button (green button) near the top right.
- Select the "Codespaces" tab.
- Click on "New codespace" to launch a new Codespace environment.
- Edit files directly within the Codespace and commit and push your changes once you're done.

## What technologies are used for this project?

This project is built with:

- Vite
- TypeScript
- React
- shadcn-ui
- Tailwind CSS

## How can I deploy this project?

Simply open [Lovable](https://lovable.dev/projects/d6b8cd81-3ed5-4585-bc99-c145dcb70ead) and click on Share -> Publish.

## Can I connect a custom domain to my Lovable project?

Yes, you can!

To connect a domain, navigate to Project > Settings > Domains and click Connect Domain.

Read more here: [Setting up a custom domain](https://docs.lovable.dev/features/custom-domain#custom-domain)

## Diabetes prediction API

This repo now includes a FastAPI service under `backend/` that loads your scikit-learn pickle file and powers the `/predict` endpoint used by the chat experience.

1. Copy your trained `diabetes_prediction_model.pkl` into `backend/models/`.
2. (Optional) Create a virtual environment.
3. Install the Python dependencies: `pip install -r backend/requirements.txt`.
4. Start the API locally: `uvicorn backend.main:app --reload --port 8000`.
5. (Optional) Create a `.env` at the repo root to override `VITE_API_BASE_URL` for the frontend, `ALLOWED_ORIGINS` for CORS, and provide `GEMINI_API_KEY` (plus `GEMINI_MODEL` / `PLAN_THRESHOLD`) so the backend can generate workout & diet plans.

For quick testing without chatting through the UI, hit:

- `GET /predict/sample-features` to inspect a valid payload.
- `POST /predict/sample` to run that payload through the model.
- `POST /predict` with your own JSON body whenever you're ready.
- High-risk cases (`>=40%`) can request Gemini-powered workout and/or diet plans via the chat, `POST /plan`, or instantly via `POST /plan/sample` (with `plan_type`/`probability` query params).
