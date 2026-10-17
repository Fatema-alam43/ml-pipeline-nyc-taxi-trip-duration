"""NYC taxi trip duration API.

Run locally:   uvicorn api.main:app --reload --port 8080
Docs:          http://localhost:8080/docs
"""

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse

from api.model import ModelService, request_to_frame, warnings_for
from api.schemas import CFG, HealthResponse, PredictionResponse, TripRequest
from src import features

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(message)s")
log = logging.getLogger("api")
DEMO_PAGE = (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup. If it fails, /health reports it instead of crashing silently."""
    try:
        app.state.model = ModelService(CFG)
    except Exception:
        log.exception("Model failed to load")
        app.state.model = None
    yield


app = FastAPI(
    title="NYC Taxi Trip Duration API",
    description="Predicts how long a New York City taxi trip will take, from information known at pickup. "
                "Model, data and limitations: see the project's model card.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """One structured (JSON) log line per request: easy to search in CloudWatch."""
    request_id = request.headers.get("x-request-id", uuid.uuid4().hex[:12])
    start = time.perf_counter()
    response = await call_next(request)
    latency_ms = round((time.perf_counter() - start) * 1000, 1)
    response.headers["x-request-id"] = request_id
    response.headers["x-latency-ms"] = str(latency_ms)
    log.info(json.dumps({"request_id": request_id, "method": request.method, "path": request.url.path,
                         "status": response.status_code, "latency_ms": latency_ms}))
    return response


@app.exception_handler(RequestValidationError)
async def clear_validation_errors(request: Request, exc: RequestValidationError):
    """Short, readable errors: which field is wrong and why."""
    errors = [{"field": ".".join(str(p) for p in err["loc"] if p != "body") or "request",
               "message": err["msg"].removeprefix("Value error, ")} for err in exc.errors()]
    return JSONResponse(status_code=422, content={"error": "Invalid trip request", "details": errors})


@app.get("/", include_in_schema=False, response_class=HTMLResponse)
def demo_page():
    """Interactive demo: click two points on a map of NYC and get a predicted trip time."""
    return DEMO_PAGE


@app.get("/health", response_model=HealthResponse)
def health():
    model = app.state.model
    if model is None:
        return JSONResponse(status_code=503, content=HealthResponse(status="unavailable",
                                                                    model_loaded=False).model_dump())
    return HealthResponse(status="ok", model_loaded=True, model_version=model.version,
                          model_sha256=model.sha256, model_source=model.source,
                          n_features=len(features.FEATURES), loaded_at=model.loaded_at)


@app.post("/predict", response_model=PredictionResponse)
def predict(trip: TripRequest):
    model = app.state.model
    if model is None:
        return JSONResponse(status_code=503, content={"error": "Model not loaded; see /health"})
    result = model.predict(request_to_frame([trip])).iloc[0]
    response = PredictionResponse(
        predicted_duration_minutes=round(float(result["seconds"]) / 60, 1),
        predicted_duration_seconds=round(float(result["seconds"]), 1),
        straight_line_distance_km=round(float(result["distance_km"]), 2),
        model_version=model.version,
        warnings=warnings_for(int(result["month"])),
    )
    log.info(json.dumps({"event": "prediction", "minutes": response.predicted_duration_minutes,
                         "distance_km": response.straight_line_distance_km}))
    return response
