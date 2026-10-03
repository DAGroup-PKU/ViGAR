"""The existing RoboTwin length-prefixed JSON/ndarray protocol, without CUDA."""

import base64
import json
import socket
import socketserver
import threading
import traceback

import numpy as np


def encode(obj):
    if isinstance(obj, np.ndarray):
        return {"__ndarray__": base64.b64encode(np.ascontiguousarray(obj).tobytes()).decode("ascii"),
                "shape": list(obj.shape), "dtype": str(obj.dtype)}
    if isinstance(obj, dict):
        return {k: encode(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [encode(v) for v in obj]
    return obj


def decode(obj):
    if isinstance(obj, dict):
        if "__ndarray__" in obj:
            return np.frombuffer(base64.b64decode(obj["__ndarray__"]),
                                 dtype=np.dtype(obj["dtype"])).reshape(obj["shape"]).copy()
        return {k: decode(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [decode(v) for v in obj]
    return obj


def recv_n(sock, size):
    data = bytearray()
    while len(data) < size:
        part = sock.recv(size - len(data))
        if not part:
            raise ConnectionError("RoboTwin RPC peer closed")
        data.extend(part)
    return bytes(data)


def receive(sock):
    size = int.from_bytes(recv_n(sock, 4), "big")
    return decode(json.loads(recv_n(sock, size)))


def send(sock, obj):
    payload = json.dumps(encode(obj)).encode("utf-8")
    sock.sendall(len(payload).to_bytes(4, "big") + payload)


class Client:
    def __init__(self, port, host="127.0.0.1", timeout=600):
        self.sock = socket.create_connection((host, int(port)), timeout=timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.lock = threading.Lock()

    def call(self, request):
        with self.lock:
            send(self.sock, request)
            response = receive(self.sock)
        if response.get("error"):
            raise RuntimeError(response["error"])
        return response

    def close(self):
        self.sock.close()


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def serve(service, port, host="127.0.0.1"):
    class Handler(socketserver.BaseRequestHandler):
        def handle(self):
            self.request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            while True:
                try:
                    request = receive(self.request)
                except (ConnectionError, OSError):
                    return
                try:
                    response = service(request)
                except Exception as exc:
                    traceback.print_exc()
                    response = {"error": f"{type(exc).__name__}: {exc}"}
                send(self.request, response)

    with Server((host, int(port)), Handler) as server:
        print(f"RPC_SERVING {host}:{port}", flush=True)
        server.serve_forever()
