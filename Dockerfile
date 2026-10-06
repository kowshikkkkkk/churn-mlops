# ============================================================
# Dockerfile
#
# Builds the SERVING image only — app/main.py + its direct
# dependencies (src/feature_engineering.py, src/preprocessing.py
# indirectly via imports). Deliberately does NOT COPY mlflow.db
# or mlruns/ into the image (see Option B decision): the MLflow
# tracking store is mounted in at `docker run` time instead, so
# a newly retrained-and-promoted model becomes visible to an
# already-running container without any rebuild — which is the
# entire point of the @production alias pattern this project
# was built around. Baking the registry in at build time would
# silently defeat that.
# ============================================================

FROM python:3.11-slim

WORKDIR /app

# Without this, Python buffers stdout inside a container (since it's
# not attached to an interactive terminal), which silently delays or
# hides print() output — including our own startup diagnostics like
# "Loaded production bundle" / "Failed to load production bundle".
ENV PYTHONUNBUFFERED=1

# Install only what serving needs — see requirements-serving.txt
# header comment for why this differs from the full dev requirements.txt
COPY requirements-serving.txt .
RUN pip install --no-cache-dir -r requirements-serving.txt

# Only the code serving actually needs — not tests/, not the
# full src/ pipeline scripts unrelated to inference
COPY app/ ./app/
COPY src/feature_engineering.py ./src/feature_engineering.py
COPY src/__init__.py ./src/__init__.py

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]