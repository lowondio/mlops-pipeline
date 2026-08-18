import pytest
from src.serve_model import PredictRequest
from pydantic import ValidationError

def test_pydantic_schema_valid():
    valid_features = [0.5] * 15
    request = PredictRequest(features=valid_features)
    assert len(request.features) == 15

def test_pydantic_schema_invalid():
    with pytest.raises(ValidationError):
        PredictRequest(features="no array")
