"""The haversine distance function, against known real-world distances."""

import numpy as np
import pytest

from src.etl import haversine_km


def test_empire_state_building_to_times_square():
    assert haversine_km(40.7484, -73.9857, 40.7580, -73.9855) == pytest.approx(1.068, abs=0.01)


def test_one_degree_of_latitude_is_111_km():
    assert haversine_km(40.0, -74.0, 41.0, -74.0) == pytest.approx(111.19, abs=0.05)


def test_one_degree_of_longitude_at_equator():
    assert haversine_km(0, 0, 0, 1) == pytest.approx(111.19, abs=0.05)


def test_same_point_is_zero():
    assert haversine_km(40.75, -73.98, 40.75, -73.98) == 0


def test_symmetric():
    assert haversine_km(40.7, -74.0, 40.8, -73.9) == pytest.approx(haversine_km(40.8, -73.9, 40.7, -74.0))


def test_works_on_arrays():
    lat = np.array([40.7484, 40.0])
    lon = np.array([-73.9857, -74.0])
    result = haversine_km(lat, lon, lat + 0.01, lon)
    assert result.shape == (2,)
    assert np.all(result > 1.0) and np.all(result < 1.2)
