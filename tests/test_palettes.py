from backend.palettes import aedat4_rgb_palette, apply_aedat4_palette, metavision_palette


class FakeVisualizer:
    def __init__(self):
        self.bg = None
        self.pos = None
        self.neg = None

    def setBackgroundColor(self, value):
        self.bg = value

    def setPositiveColor(self, value):
        self.pos = value

    def setNegativeColor(self, value):
        self.neg = value


class FakeColorPalette:
    Dark = "dark"
    Light = "light"
    CoolWarm = "cool-warm"
    Gray = "gray"


def test_aedat4_rgb_palette_returns_named_palette():
    assert aedat4_rgb_palette("Light") == {
        "bg": (255, 255, 255),
        "pos": (64, 126, 200),
        "neg": (30, 37, 52),
    }


def test_aedat4_rgb_palettes_match_metavision_sdk_colors():
    assert aedat4_rgb_palette("Dark") == {
        "bg": (30, 37, 52),
        "pos": (255, 255, 255),
        "neg": (64, 126, 200),
    }
    assert aedat4_rgb_palette("Light") == {
        "bg": (255, 255, 255),
        "pos": (64, 126, 200),
        "neg": (30, 37, 52),
    }
    assert aedat4_rgb_palette("CoolWarm") == {
        "bg": (217, 224, 237),
        "pos": (255, 113, 117),
        "neg": (87, 123, 198),
    }
    assert aedat4_rgb_palette("Gray") == {
        "bg": (128, 128, 128),
        "pos": (255, 255, 255),
        "neg": (0, 0, 0),
    }


def test_aedat4_rgb_palette_falls_back_to_dark():
    assert aedat4_rgb_palette("Unknown") == aedat4_rgb_palette("Dark")


def test_apply_aedat4_palette_updates_visualizer():
    visualizer = FakeVisualizer()

    apply_aedat4_palette(visualizer, "Gray")

    assert visualizer.bg == (128, 128, 128)
    assert visualizer.pos == (255, 255, 255)
    assert visualizer.neg == (0, 0, 0)


def test_metavision_palette_maps_names_and_falls_back():
    assert metavision_palette(FakeColorPalette, "CoolWarm") == "cool-warm"
    assert metavision_palette(FakeColorPalette, "Unknown") == "dark"
