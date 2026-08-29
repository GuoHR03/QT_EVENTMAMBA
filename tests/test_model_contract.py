import pytest

from backend.model_contract import (
    EVENT_SHAPE,
    MODEL_INPUT_SHAPE,
    MODE_CENTER,
    MODE_ELLIPSE,
    SUPPORTED_MODEL_MODES,
    get_model_spec,
    is_supported_model_mode,
)


def test_model_contract_centralizes_shapes_and_output_schemas():
    assert EVENT_SHAPE == (1024, 3)
    assert MODEL_INPUT_SHAPE == (1, 3, 1024)
    assert SUPPORTED_MODEL_MODES == (MODE_CENTER, MODE_ELLIPSE)
    assert get_model_spec(MODE_CENTER).output_size == 2
    assert get_model_spec(MODE_ELLIPSE).output_size == 5
    assert get_model_spec(MODE_ELLIPSE).requires_matrix is True


def test_model_mode_lookup_normalizes_user_input():
    assert get_model_spec(" ELLIPSE ").mode == MODE_ELLIPSE
    assert is_supported_model_mode(" CENTER ") is True
    assert is_supported_model_mode("unknown") is False


def test_unknown_model_mode_has_one_canonical_validation_error():
    with pytest.raises(ValueError, match="Unsupported prediction mode"):
        get_model_spec("unknown")
