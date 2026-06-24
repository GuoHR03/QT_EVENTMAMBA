AEDAT4_RGB_PALETTES = {
    "Dark": {
        "bg": (30, 37, 52),
        "pos": (255, 255, 255),
        "neg": (64, 126, 200),
    },
    "Light": {
        "bg": (255, 255, 255),
        "pos": (64, 126, 200),
        "neg": (30, 37, 52),
    },
    "CoolWarm": {
        "bg": (217, 224, 237),
        "pos": (255, 113, 117),
        "neg": (87, 123, 198),
    },
    "Gray": {
        "bg": (128, 128, 128),
        "pos": (255, 255, 255),
        "neg": (0, 0, 0),
    },
}


def aedat4_rgb_palette(name):
    return AEDAT4_RGB_PALETTES.get(name, AEDAT4_RGB_PALETTES["Dark"])


def apply_aedat4_palette(visualizer, name):
    rgb = aedat4_rgb_palette(name)
    visualizer.setBackgroundColor(rgb["bg"])
    visualizer.setPositiveColor(rgb["pos"])
    visualizer.setNegativeColor(rgb["neg"])


def metavision_palette(color_palette, name):
    palette_map = {
        "Dark": color_palette.Dark,
        "Light": color_palette.Light,
        "CoolWarm": color_palette.CoolWarm,
        "Gray": color_palette.Gray,
    }
    return palette_map.get(name, color_palette.Dark)
