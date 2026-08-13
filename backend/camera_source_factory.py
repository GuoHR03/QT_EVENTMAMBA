"""Backward-compatible exports for the split source construction modules."""

from backend.event_source import Aedat4Source, H5Source, MetavisionSource, SourceMetadata
from backend.renderer_factory import (
    DynamicMetavisionFrameGenerator,
    MetavisionFrameRenderer,
    create_metavision_frame_generator,
    create_metavision_renderer,
)
from backend.source_factory import (
    create_aedat4_source,
    create_event_source,
    create_h5_source,
    create_metavision_source,
)
from backend.source_metadata import (
    SOURCE_AEDAT4,
    SOURCE_H5,
    SOURCE_METAVISION,
    classify_input_source,
)


__all__ = [
    "Aedat4Source",
    "DynamicMetavisionFrameGenerator",
    "H5Source",
    "MetavisionFrameRenderer",
    "MetavisionSource",
    "SOURCE_AEDAT4",
    "SOURCE_H5",
    "SOURCE_METAVISION",
    "SourceMetadata",
    "classify_input_source",
    "create_aedat4_source",
    "create_event_source",
    "create_h5_source",
    "create_metavision_frame_generator",
    "create_metavision_renderer",
    "create_metavision_source",
]
