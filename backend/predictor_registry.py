from dataclasses import dataclass


@dataclass(frozen=True)
class PredictorSpec:
    """Describe how one prediction mode constructs its runtime predictor."""

    mode: str
    factory: object

    def __post_init__(self):
        mode = str(self.mode or "").strip().lower()
        if not mode:
            raise ValueError("predictor mode is required")
        if not callable(self.factory):
            raise TypeError("predictor factory must be callable")
        object.__setattr__(self, "mode", mode)

    def create(self, weights_path, device):
        return self.factory(weights_path, device)


class PredictorRegistry:
    """Map prediction modes to replaceable predictor implementations."""

    def __init__(self, specs=None):
        self._specs = {}
        for spec in specs or ():
            self.register(spec)

    @property
    def modes(self):
        return tuple(self._specs)

    def supports(self, mode):
        return self._normalize_mode(mode) in self._specs

    def register(self, spec, replace=False):
        if not isinstance(spec, PredictorSpec):
            raise TypeError("spec must be a PredictorSpec")
        if spec.mode in self._specs and not replace:
            raise ValueError(f"predictor mode is already registered: {spec.mode}")
        self._specs[spec.mode] = spec
        return spec

    def create(self, mode, weights_path, device):
        normalized_mode = self._normalize_mode(mode)
        spec = self._specs.get(normalized_mode)
        if spec is None:
            supported = ", ".join(self.modes) or "none"
            raise ValueError(
                f"Unsupported prediction mode: {normalized_mode}; supported: {supported}"
            )
        return spec.create(weights_path, device)

    @staticmethod
    def _normalize_mode(mode):
        return str(mode or "").strip().lower()
