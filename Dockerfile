FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application.
COPY arbiter ./arbiter
COPY ui ./ui
COPY examples ./examples
COPY tests ./tests
COPY README.md pytest.ini ./

# Persisted audit trail (SQLite + JSON) lives here.
RUN mkdir -p /app/data
EXPOSE 8000 8501

# Default: run the FastAPI service. The `ui` service overrides this command.
CMD ["uvicorn", "arbiter.api:app", "--host", "0.0.0.0", "--port", "8000"]
