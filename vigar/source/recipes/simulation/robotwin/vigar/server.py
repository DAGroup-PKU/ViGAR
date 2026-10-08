"""Internal native ViGAR HTTP server; start with serve.sh (one or more GPUs)."""

import argparse
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path


os.environ.setdefault("COSMOS_TRAINING", "1")


def main():
    import torch
    import torch.distributed as dist
    from cosmos_framework.inference.common.init import _init_log_console

    from recipes.simulation.robotwin.common.transport import MAX_MESSAGE_BYTES, dumps, loads
    from recipes.simulation.robotwin.vigar.policy import ViGARSimulationPolicy

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, help="Required for initial bundles; trained bundles carry their recipe")
    parser.add_argument("--weights", choices=["regular", "ema"], default="ema")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18765)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decode-video", action="store_true")
    args = parser.parse_args()
    _init_log_console()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    dist.init_process_group("nccl")
    control = dist.new_group(backend="gloo")
    try:
        policy = ViGARSimulationPolicy(
            args.checkpoint, args.recipe, weights=args.weights, output=args.output, decode_video=args.decode_video
        )

        def execute(request):
            # Every rank participates in FSDP even though only rank 0 handles HTTP.
            result, error = None, None
            try:
                result = policy.infer(request)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            errors = [None] * dist.get_world_size()
            dist.all_gather_object(errors, error, group=control)
            if any(errors):
                raise RuntimeError(str(errors))
            return result

        if dist.get_rank() != 0:
            while True:
                message = [None]
                dist.broadcast_object_list(message, src=0, group=control)
                if message[0] is None:
                    break
                try:
                    execute(message[0])
                except RuntimeError as error:
                    print(error, flush=True)
            return

        class Handler(BaseHTTPRequestHandler):
            def respond(self, status, value):
                body = dumps(value)
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path != "/health":
                    self.respond(404, dict(error="Unknown endpoint"))
                    return
                self.respond(
                    200,
                    dict(
                        ok=True,
                        model="ViGAR",
                        action_horizon=48,
                        action_dim=49,
                        weights=args.weights,
                        checkpoint=str(args.checkpoint.resolve()),
                        sampling={
                            key: policy.settings["train"].get(field, default)
                            for key, field, default in (
                                ("num_steps", "gen_num_steps", 5),
                                ("guidance", "gen_guidance", 3.0),
                                ("shift", "gen_shift", 5.0),
                            )
                        },
                        world_size=dist.get_world_size(),
                        output=str(args.output.resolve()),
                        control_modes=["ee", "qpos"]
                        if policy.settings["data"].get("supervise_arm_head_torso", True)
                        else ["ee"],
                    ),
                )

            def do_POST(self):
                if self.path != "/infer":
                    self.respond(404, dict(error="Unknown endpoint"))
                    return
                try:
                    size = int(self.headers.get("Content-Length", "0"))
                    if not 0 < size <= MAX_MESSAGE_BYTES:
                        raise ValueError("Invalid request size")
                    request = loads(self.rfile.read(size))
                    policy.processor.raw_sample(request)  # Reject bad inputs before any collective.
                except Exception as error:
                    self.respond(400, dict(error=f"{type(error).__name__}: {error}"))
                    return
                dist.broadcast_object_list([request], src=0, group=control)
                try:
                    self.respond(200, execute(request))
                except Exception as error:
                    self.respond(500, dict(error=f"{type(error).__name__}: {error}"))

        server = HTTPServer((args.host, args.port), Handler)
        server.timeout = 1
        print(f"ViGAR ready at http://{args.host}:{args.port}", flush=True)
        try:
            server.serve_forever()
        finally:
            server.server_close()
            dist.broadcast_object_list([None], src=0, group=control)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
