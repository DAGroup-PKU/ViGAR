"""Small JSON/NumPy HTTP protocol shared by independent simulator/model processes."""

import base64
import json
import urllib.request

import numpy as np


MAX_MESSAGE_BYTES = 32 * 1024 * 1024


def encode(value):
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "buif":
            raise ValueError("Only numeric and boolean arrays can cross the policy protocol")
        return dict(
            __array__=base64.b64encode(np.ascontiguousarray(value).tobytes()).decode("ascii"),
            dtype=value.dtype.str,
            shape=list(value.shape),
        )
    if isinstance(value, dict):
        return {k: encode(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [encode(v) for v in value]
    return value


def decode(value):
    if isinstance(value, dict):
        if "__array__" in value:
            dtype = np.dtype(value["dtype"])
            shape = tuple(value["shape"])
            if dtype.kind not in "buif" or any(not isinstance(s, int) or s < 0 for s in shape):
                raise ValueError("Invalid array dtype/shape")
            data = base64.b64decode(value["__array__"], validate=True)
            if len(data) > MAX_MESSAGE_BYTES:
                raise ValueError("Array exceeds message limit")
            return np.frombuffer(data, dtype=dtype).reshape(shape).copy()
        return {k: decode(v) for k, v in value.items()}
    if isinstance(value, list):
        return [decode(v) for v in value]
    return value


def dumps(value):
    payload = json.dumps(encode(value), allow_nan=False).encode()
    if len(payload) > MAX_MESSAGE_BYTES:
        raise ValueError("Policy message exceeds 32 MiB")
    return payload


def loads(payload):
    if len(payload) > MAX_MESSAGE_BYTES:
        raise ValueError("Policy message exceeds 32 MiB")
    return decode(json.loads(payload))


class PolicyClient:
    def __init__(self, url="http://127.0.0.1:18765", timeout=600):
        self.url, self.timeout = url.rstrip("/"), timeout

    def call(self, path, value=None):
        request = urllib.request.Request(
            self.url + path, data=None if value is None else dumps(value), headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            result = loads(response.read(MAX_MESSAGE_BYTES + 1))
        if result.get("error"):
            raise RuntimeError(result["error"])
        return result
