from app.ui_status import source_display_name


def test_source_display_name_covers_supported_event_files():
    assert source_display_name("") == "实时输入"
    assert source_display_name("recording.RAW") == "RAW 文件"
    assert source_display_name("events.h5") == "H5 文件"
    assert source_display_name("events.hdf5") == "H5 文件"
    assert source_display_name("events.AEDAT4") == "AEDAT4 文件"


def test_source_display_name_falls_back_for_other_files():
    assert source_display_name("events.bin") == "离线文件"
