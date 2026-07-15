from backend.source_metadata import aedat4_resolution, aedat4_time_range


def test_aedat4_resolution_supports_object_shape():
    class Resolution:
        width = 640
        height = 480

    class Reader:
        def getEventResolution(self):
            return Resolution()

    assert aedat4_resolution(Reader()) == (640, 480)


def test_aedat4_time_range_supports_start_plus_duration_fallback():
    class Reader:
        def getStartTimeUs(self):
            return 1000

        def getDurationUs(self):
            return 5000

    assert aedat4_time_range(Reader()) == (1000, 6000)


def test_aedat4_time_range_supports_range_object_attributes():
    class TimeRange:
        start = 25
        end = 75

    class Reader:
        def getTimeRange(self):
            return TimeRange()

    assert aedat4_time_range(Reader()) == (25, 75)
