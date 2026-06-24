from backend.camera_source_factory import (
    SOURCE_AEDAT4,
    SOURCE_H5,
    SOURCE_METAVISION,
    classify_input_source,
)


def test_classify_input_source_detects_aedat4():
    assert classify_input_source("sample.AEDAT4") == SOURCE_AEDAT4


def test_classify_input_source_detects_h5_variants():
    assert classify_input_source("sample.h5") == SOURCE_H5
    assert classify_input_source("sample.HDF5") == SOURCE_H5


def test_classify_input_source_defaults_to_metavision():
    assert classify_input_source("") == SOURCE_METAVISION
    assert classify_input_source("recording.raw") == SOURCE_METAVISION
