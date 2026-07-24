# Deployment

## What can go where

This project has two runtime shapes with very different hosting needs:

| Component | Nature | Good hosts |
|---|---|---|
| **FastAPI API** (`arbiter/api.py`) | Stateless request/response | **Vercel** (serverless), Render, Railway, Fly.io |
| **Streamlit UI** (`ui/streamlit_app.py`) | Long-running server + websockets | Streamlit Community Cloud, Render, Railway, Fly.io |
| **Ollama** (completeness critic) | Persistent container + model weights | Fly.io, Railway, a VM — **not** serverless |
| **Full stack together** | 3 services + volume | `docker compose up` on any Docker host |

> **Streamlit and Ollama cannot run on Vercel.** Vercel is a serverless platform:
> functions are stateless and time-limited, with no persistent process or
> filesystem. That's fine for the API but incompatible with Streamlit's
> long-lived server and Ollama's model runtime.

---

## Deploy the full UI to Streamlit Community Cloud (recommended, free, keyless)

This is the easiest way to put the **Verdict Explorer UI** online. Streamlit
Community Cloud runs the app straight from this GitHub repo — no keys, no CLI, no
credit card. The repo is already configured for it (`ui/streamlit_app.py` entry,
root `requirements.txt`, `.streamlit/config.toml` theme).

1. Push these files to your default branch (already done if you merged the PR).
2. Go to <https://share.streamlit.io> and sign in with GitHub (authorise access
   to the `New-Project` repo — public repos work out of the box).
3. Click **Create app → Deploy a public app from GitHub** and set:
   - **Repository:** `AdityaRungta2048/New-Project`
   - **Branch:** `main`
   - **Main file path:** `ui/streamlit_app.py`
   - *(Advanced → Python version: 3.11 or newer.)*
4. Click **Deploy**. First build takes a few minutes while dependencies install;
   after that your app is live at
   `https://<your-app-name>.streamlit.app`.

It runs on the deterministic **mock backend** by default, so every view works
with no API keys. To route critics through real models, open
**Manage app → Settings → Secrets** and add:

```toml
ARBITER_ACCURACY_BACKEND = "openai"
ARBITER_LOGIC_BACKEND = "anthropic"
OPENAI_API_KEY = "sk-..."
ANTHROPIC_API_KEY = "sk-ant-..."
```

(Streamlit Cloud exposes secrets as environment variables, which the app reads.
Leave the completeness critic on `mock` — Ollama has no home on Community Cloud —
or point `OLLAMA_HOST` at an external Ollama instance.)

> The SQLite audit trail is ephemeral on Community Cloud (the container's disk
> resets on reboot), which is fine for a live demo. For a durable trail, use a
> Docker host with a mounted volume (below).

---

## Deploy the API to Vercel

The repo is already configured (`vercel.json`, `api/index.py`,
`api/requirements.txt`, `.vercelignore`). On Vercel the API runs with the
deterministic **mock backend** by default (no keys needed) and stores its audit
trail in `/tmp` (ephemeral — see the note below).

### Option A — Vercel dashboard (no CLI)

1. Go to <https://vercel.com/new> and **Import** the GitHub repo
   `AdityaRungta2048/New-Project`.
2. Framework preset: **Other** (the included `vercel.json` handles the Python
   build — leave build/output settings empty).
3. Click **Deploy**.
4. When it finishes, open:
   - `https://<your-app>.vercel.app/` — service metadata
   - `https://<your-app>.vercel.app/docs` — interactive OpenAPI docs
   - `POST https://<your-app>.vercel.app/v1/arbitrate` — the arbitration endpoint

### Option B — Vercel CLI

```bash
npm i -g vercel
vercel        # first run links the project
vercel --prod # production deployment
```

### Enable real models (optional)

Add these in **Vercel → Project → Settings → Environment Variables**, then add
`instructor`, `openai`, `anthropic` to `api/requirements.txt`:

```
ARBITER_ACCURACY_BACKEND = openai
ARBITER_LOGIC_BACKEND     = anthropic
OPENAI_API_KEY            = sk-...
ANTHROPIC_API_KEY         = sk-ant-...
```

(The completeness critic's Ollama backend has no serverless host, so leave it on
`mock` for Vercel, or point `OLLAMA_HOST` at an external Ollama instance.)

> **Ephemeral storage note:** Vercel's only writable path is `/tmp`, which is not
> shared across cold starts. `POST /v1/arbitrate` and `/health` work perfectly,
> but `GET /v1/arbitrations/{id}`, the list endpoint, and `/v1/analytics` only
> reflect the current warm instance. For a durable audit trail, deploy the API on
> a host with a persistent disk (below) or point `ARBITER_DB_PATH` at an external
> database volume.

---

## Deploy the full stack (API + UI + Ollama)

Any Docker host runs the complete system as-is:

```bash
docker compose up --build
# API -> :8000/docs   UI -> :8501   Ollama -> :11434
```

Render / Railway / Fly.io can each build from the included `Dockerfile`. For the
UI service, set the start command to:

```bash
streamlit run ui/streamlit_app.py --server.port $PORT --server.address 0.0.0.0
```

and point `ARBITER_COMPLETENESS_BACKEND` / `OLLAMA_HOST` at your Ollama instance
(or leave everything on `mock` for a keyless demo).
