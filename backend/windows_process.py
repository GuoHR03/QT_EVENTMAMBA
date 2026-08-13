from backend.backend_process import BackendProcess


def _windows_backend_arguments(
    center_model_path,
    ellipse_model_path,
    ellipse_matrix_path,
    custom_op_library,
    initial_mode,
    port,
    instance_nonce=None,
):
    arguments = [
        "--center-model",
        center_model_path,
        "--ellipse-model",
        ellipse_model_path,
        "--ellipse-matrix",
        ellipse_matrix_path,
        "--custom-op-library",
        custom_op_library,
        "--initial-mode",
        initial_mode,
        "--port",
        str(port),
    ]
    if instance_nonce is not None:
        arguments.extend(("--instance-nonce", str(instance_nonce)))
    return arguments


def build_windows_backend_command(
    python_executable,
    backend_script,
    center_model_path,
    ellipse_model_path,
    ellipse_matrix_path,
    custom_op_library,
    initial_mode,
    port,
    instance_nonce=None,
):
    return [
        python_executable,
        backend_script,
        *_windows_backend_arguments(
            center_model_path,
            ellipse_model_path,
            ellipse_matrix_path,
            custom_op_library,
            initial_mode,
            port,
            instance_nonce,
        ),
    ]


def build_windows_backend_executable_command(
    backend_executable,
    center_model_path,
    ellipse_model_path,
    ellipse_matrix_path,
    custom_op_library,
    initial_mode,
    port,
    instance_nonce=None,
):
    return [
        backend_executable,
        *_windows_backend_arguments(
            center_model_path,
            ellipse_model_path,
            ellipse_matrix_path,
            custom_op_library,
            initial_mode,
            port,
            instance_nonce,
        ),
    ]


class WindowsBackendProcess(BackendProcess):
    """Local backend process; intentionally does not kill other Python jobs."""
