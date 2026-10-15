"""Servidor independiente de Qwen3-VL-8B.

Se lanza aparte del resto del programa (otro proceso, otro entorno
virtual, incluso otra máquina). Los clientes solo envían prompts con
imágenes por HTTP, con un subconjunto de la API OpenAI:

    GET  /health
    GET  /v1/models
    POST /v1/chat/completions   (imágenes como data URI base64)

Uso:
    python -m human_reid_ros.qwen3vl_server --port 8000
"""
import argparse
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .vlms.common import (
    DEFAULT_MAX_PIXELS,
    DEFAULT_MODEL_PATH,
    REPO_ID,
    from_openai_messages,
)

MAX_BODY_BYTES = 256 * 1024 * 1024

log = logging.getLogger("qwen3vl_server")


def make_server(engine, host="127.0.0.1", port=8000, model_name=REPO_ID):
    """Crea el servidor HTTP. ``engine`` solo necesita ``chat(messages, max_new_tokens)``."""

    # La GPU atiende una petición cada vez.
    gpu_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):

        def log_message(self, fmt, *args):
            log.info("%s - %s", self.address_string(), fmt % args)

        def _send_json(self, status, payload):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _error(self, status, message):
            self._send_json(status, {"error": {"message": message}})

        def do_GET(self):
            if self.path == "/health":
                self._send_json(200, {"status": "ok", "model": model_name})
            elif self.path == "/v1/models":
                self._send_json(200, {
                    "object": "list",
                    "data": [{"id": model_name, "object": "model"}],
                })
            else:
                self._error(404, f"Ruta desconocida: {self.path}")

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                self._error(404, f"Ruta desconocida: {self.path}")
                return

            try:
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= MAX_BODY_BYTES:
                    raise ValueError("Content-Length inválido o demasiado grande.")
                request = json.loads(self.rfile.read(length))
                messages = from_openai_messages(request["messages"])
                max_new_tokens = int(request.get("max_tokens") or 512)
            except (ValueError, KeyError, TypeError) as exc:
                self._error(400, f"Petición inválida: {exc}")
                return

            try:
                with gpu_lock:
                    text = engine.chat(messages, max_new_tokens=max_new_tokens)
            except Exception as exc:  # noqa: BLE001
                log.exception("Error durante la inferencia")
                self._error(500, f"Error de inferencia: {exc}")
                return

            self._send_json(200, {
                "object": "chat.completion",
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }],
            })

    return ThreadingHTTPServer((host, port), Handler)


def main(args=None):

    parser = argparse.ArgumentParser(
        description="Servidor independiente de Qwen3-VL-8B (sin descargas)."
    )
    parser.add_argument("--host", default="127.0.0.1",
                        help="Usa 0.0.0.0 para aceptar otras máquinas.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH,
                        help="Carpeta con los pesos ya instalados.")
    parser.add_argument("--quantization", choices=("4bit", "8bit", "none"),
                        default="4bit")
    parser.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS)
    opts = parser.parse_args(args)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    # Import diferido: torch solo se carga al arrancar el servidor.
    from .vlms.qwen3vl_engine import Qwen3VLEngine

    engine = Qwen3VLEngine(opts.model_path, opts.quantization, opts.max_pixels)
    server = make_server(engine, opts.host, opts.port)

    log.info("Servidor listo en http://%s:%d", opts.host, opts.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
