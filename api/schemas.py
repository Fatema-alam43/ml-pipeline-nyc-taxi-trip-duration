"""Request and response models. Pydantic validates every request before it reaches the model."""

from datetime import datetime
from typing import Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.config import load_config

CFG = load_config()
BOX = CFG["cleaning"]["nyc_bbox"]
NYC_TZ = ZoneInfo("America/New_York")
EXAMPLE = {
    "pickup_latitude": 40.7580, "pickup_longitude": -73.9855,          # Times Square
    "dropoff_latitude": 40.6413, "dropoff_longitude": -73.7781,        # JFK Airport
    "pickup_datetime": "2016-06-15T17:30:00",
    "passenger_count": 1,
}


class TripRequest(BaseModel):
    """A trip that is about to start. Every field is known before the trip happens."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [EXAMPLE]})

    pickup_latitude: float = Field(description="Pickup latitude (decimal degrees)")
    pickup_longitude: float = Field(description="Pickup longitude (decimal degrees)")
    dropoff_latitude: float = Field(description="Dropoff latitude (decimal degrees)")
    dropoff_longitude: float = Field(description="Dropoff longitude (decimal degrees)")
    pickup_datetime: datetime = Field(description="Pickup time. Without a timezone it is read as New York time")
    passenger_count: int = Field(1, ge=1, le=6, description="Passengers (1 to 6)")
    vendor_id: Literal[1, 2] = Field(1, description="Taxi company (1 or 2)")
    store_and_fwd_flag: Literal["N", "Y"] = Field("N", description="Trip stored offline before sending")

    @field_validator("pickup_latitude", "dropoff_latitude")
    @classmethod
    def latitude_in_nyc(cls, value: float, info) -> float:
        if not BOX["lat_min"] <= value <= BOX["lat_max"]:
            raise ValueError(f"{info.field_name} {value} is outside New York City "
                             f"(allowed {BOX['lat_min']} to {BOX['lat_max']})")
        return value

    @field_validator("pickup_longitude", "dropoff_longitude")
    @classmethod
    def longitude_in_nyc(cls, value: float, info) -> float:
        if not BOX["lon_min"] <= value <= BOX["lon_max"]:
            raise ValueError(f"{info.field_name} {value} is outside New York City "
                             f"(allowed {BOX['lon_min']} to {BOX['lon_max']})")
        return value

    @field_validator("pickup_datetime")
    @classmethod
    def to_new_york_time(cls, value: datetime) -> datetime:
        """The model learned New York local time patterns, so convert any timezone to it."""
        if value.tzinfo is not None:
            value = value.astimezone(NYC_TZ).replace(tzinfo=None)
        return value

    @model_validator(mode="after")
    def different_points(self):
        if (self.pickup_latitude, self.pickup_longitude) == (self.dropoff_latitude, self.dropoff_longitude):
            raise ValueError("pickup and dropoff are the same point; a trip needs two different locations")
        return self


class PredictionResponse(BaseModel):
    predicted_duration_minutes: float = Field(description="Predicted trip duration in minutes")
    predicted_duration_seconds: float
    straight_line_distance_km: float
    model_version: str
    warnings: list[str] = Field(default_factory=list,
                                description="Reasons the prediction may be less reliable")


class HealthResponse(BaseModel):
    status: Literal["ok", "unavailable"]
    model_loaded: bool
    model_version: str | None = None
    model_sha256: str | None = None
    model_source: str | None = None
    n_features: int | None = None
    loaded_at: str | None = None
