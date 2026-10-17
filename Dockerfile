# Prediction API image. The same image runs locally, on ECS/Fargate, and on AWS Lambda.
#   docker build --build-arg PYTHON_VERSION=$(cat .python-version) -t nyc-taxi-api .
#   docker run -p 8080:8080 -v "$(pwd)/models:/models:ro" -e MODEL_PATH=/models/lgbm_improved.txt nyc-taxi-api
ARG PYTHON_VERSION=3.12
FROM public.ecr.aws/docker/library/python:${PYTHON_VERSION}-slim

# AWS Lambda Web Adapter: on Lambda it turns Lambda events into HTTP requests for the app.
# Everywhere else it does nothing, so the image stays a normal web server.
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:1.0.1 /lambda-adapter /opt/extensions/lambda-adapter

# LightGBM needs the OpenMP runtime library
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    AWS_LWA_PORT=8080 \
    AWS_LWA_READINESS_CHECK_PATH=/health

WORKDIR /app
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Only what the API needs: config, the shared feature code, and the API
COPY config.yaml .
COPY src/__init__.py src/config.py src/etl.py src/features.py src/
COPY api/ api/

# Run as a non-root user
RUN useradd --create-home --uid 1000 appuser
USER appuser

EXPOSE 8080
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
