# Build context is the project root: docker build -f service/Dockerfile -t retail-demand-api:1.0.0 .
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    MODEL_STORE=/app/model_store \
    HISTORY_PATH=/app/data/serving_history.parquet

WORKDIR /app

# lightgbm's shared library links against libgomp, which python:*-slim does not ship
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY service/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY src/demand_features.py src/demand_eval.py src/demand_model.py /app/
COPY service/app /app/app
COPY service/model_store /app/model_store
COPY service/data/serving_history.parquet /app/data/serving_history.parquet

RUN useradd --create-home --uid 10001 forecast && chown -R forecast:forecast /app
USER forecast

# build-time smoke check as the runtime user: fails the build if a system library is missing
# or the baked registry copy cannot resolve the model
RUN python -c "\
import json, mlflow, lightgbm, pandas, numpy, pyarrow;\
manifest = json.load(open('/app/model_store/manifest.json'));\
mlflow.set_tracking_uri(manifest['container_tracking_uri']);\
model = mlflow.pyfunc.load_model(manifest['model_uri']);\
print('smoke ok', lightgbm.__version__, pandas.__version__, numpy.__version__, manifest['model_uri'])"

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/healthz').read()"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
