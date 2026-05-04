#Windows端
import queue
import zmq
from PyQt6.QtCore import QThread, pyqtSignal

class NetworkThread(QThread):
    result_signal = pyqtSignal(str, object)

    def __init__(self, input_queue, host="127.0.0.1", port=5555):
        super().__init__()
        self.input_queue = input_queue
        self.running = True
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{host}:{port}")

    def run(self):
        """与linux通信"""
        while self.running:
            try:
                data = None
                try:
                    while True:
                        data = self.input_queue.get_nowait()
                except queue.Empty:
                    pass

                if data is None:
                    data = self.input_queue.get(timeout=1.0)

                timestamp = 0
                if isinstance(data, dict) and "timestamp" in data:
                    timestamp = data["timestamp"]

                self.socket.send_pyobj(data)
                result = self.socket.recv_string()
                self.result_signal.emit(result, timestamp)

            except queue.Empty:
                continue
            except Exception as e:
                    print(f"通信异常")

    def stop(self):
        self.running = False
        self.socket.close(linger=0)
        self.context.term()
