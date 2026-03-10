"""Raw light-curve acquisition sources."""
from triceratops.lightcurve.sources.array import ArrayRawSource
from triceratops.lightcurve.sources.file import FileRawSource
from triceratops.lightcurve.sources.lightkurve import LightkurveRawSource

__all__ = ["ArrayRawSource", "FileRawSource", "LightkurveRawSource"]
