import sys
import traceback
from pathlib import Path

from .bootstrap import app_resource_path, configure_runtime


def run_packaged_runtime_smoke_test():
    """Exercise optional native runtimes without opening the interactive UI."""
    configure_runtime(__file__)

    import dv_processing
    import h5py
    import metavision_core.event_io
    import metavision_sdk_core
    import metavision_sdk_cv

    from backend.noise_filter import NoiseFilterPipeline
    from backend.renderer_factory import create_metavision_renderer

    required_modules = (
        dv_processing,
        h5py,
        metavision_core.event_io,
        metavision_sdk_core,
        metavision_sdk_cv,
    )
    if getattr(sys, "frozen", False):
        bundle_root = Path(app_resource_path(".")).resolve()
        for module in required_modules:
            module_file = getattr(module, "__file__", "")
            module_path = Path(module_file).resolve() if module_file else None
            if (
                module_path is not None
                and module_path != bundle_root
                and bundle_root not in module_path.parents
            ):
                raise RuntimeError(
                    f"Packaged module escaped bundle: {module.__name__}: {module_file}"
                )

    renderer = create_metavision_renderer(
        640,
        480,
        30,
        "Dark",
        lambda *_args: None,
    )
    renderer.close()
    for filter_name in ("activity", "trail", "stc", "anti_flicker"):
        pipeline = NoiseFilterPipeline(filter_name, 10000)
        pipeline.initialize(640, 480)
        if not pipeline.enabled:
            raise RuntimeError(f"Noise filter failed to initialize: {filter_name}")


def runtime_smoke_exit_code(argv=None, runner=None):
    argv = sys.argv if argv is None else argv
    if "--runtime-smoke-test" not in argv:
        return None

    runner = run_packaged_runtime_smoke_test if runner is None else runner
    try:
        runner()
    except Exception:
        traceback.print_exc()
        return 1
    return 0
