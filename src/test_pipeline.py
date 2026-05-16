import pytest
from src.serve_model import PredictRequest
from pydantic import ValidationError

def test_pydantic_schema_valid():
    # Проверяем, что схема принимает правильный массив из 15 чисел
    valid_features = [0.5] * 15
    request = PredictRequest(features=valid_features)
    assert len(request.features) == 15

def test_pydantic_schema_invalid():
    # Проверяем, что схема не принимает строку вместо списка
    with pytest.raises(ValidationError):
        PredictRequest(features="это текст, а не массив")