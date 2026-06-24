import os
import time

import zmq

from backend.inference_runtime import decode_backend_log


def wait_for_backend_ready_with_log(
    host,
    port,
    timeout_s,
    backend_process,
    log_path,
    poll_interval_s=0.2,
):
    endpoint = f"tcp://{host}:{port}"
    deadline = time.time() + timeout_s
    context = zmq.Context()
    try:
        while time.time() < deadline:
            if backend_process is not None and backend_process.poll() is not None:
                details = read_backend_log_tail(log_path)
                if details:
                    raise RuntimeError(f"WSL 推理服务启动失败，进程已退出。\n后端日志：\n{details}")
                raise RuntimeError("WSL 推理服务启动失败，进程已退出")

            socket = context.socket(zmq.REQ)
            socket.setsockopt(zmq.LINGER, 0)
            socket.setsockopt(zmq.RCVTIMEO, int(poll_interval_s * 1000))
            socket.setsockopt(zmq.SNDTIMEO, int(poll_interval_s * 1000))
            try:
                socket.connect(endpoint)
                socket.send_pyobj({"msg_type": "PING"})
                reply = socket.recv_string()
                if reply == "READY":
                    return
            except zmq.Again:
                time.sleep(poll_interval_s)
            except zmq.ZMQError:
                time.sleep(poll_interval_s)
            finally:
                socket.close(linger=0)

        details = read_backend_log_tail(log_path)
        if details:
            raise TimeoutError(f"等待 WSL 推理服务就绪超时。\n后端日志：\n{details}")
        raise TimeoutError("等待 WSL 推理服务就绪超时")
    finally:
        context.term()


def read_backend_log_tail(log_path, max_chars=4000):
    try:
        if not os.path.exists(log_path):
            return ""
        with open(log_path, "rb") as handle:
            raw = handle.read()
        content = decode_backend_log(raw)
        return content[-max_chars:].strip()
    except Exception:
        return ""
