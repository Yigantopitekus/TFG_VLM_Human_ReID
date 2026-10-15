"""Cliente HTTP del servidor Qwen3-VL (solo biblioteca estándar).

Habla el subconjunto de la API OpenAI ``/v1/chat/completions``, así que
también funciona contra otros servidores compatibles (p. ej. vLLM).
"""
import json
import time
import urllib.error
import urllib.request

from .common import DEFAULT_SERVER_URL, REPO_ID, to_openai_messages


class RemoteChat:

    def __init__(self, server_url=DEFAULT_SERVER_URL, model=REPO_ID,
                 timeout=300.0):
        self.server_url = server_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def is_ready(self, timeout=2.0):
        try:
            with urllib.request.urlopen(
                self.server_url + "/health", timeout=timeout
            ) as response:
                return response.status == 200
        except (OSError, urllib.error.URLError):
            return False

    def wait_until_ready(self, wait_seconds=0.0):
        deadline = time.monotonic() + wait_seconds
        while True:
            if self.is_ready():
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(1.0)

    def chat(self, messages, max_new_tokens=512):

        body = json.dumps({
            "model": self.model,
            "messages": to_openai_messages(messages),
            "max_tokens": max_new_tokens,
            "temperature": 0,
        }).encode("utf-8")

        request = urllib.request.Request(
            self.server_url + "/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as r:
                data = json.load(r)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"El servidor VLM respondió {exc.code}: {detail}"
            ) from exc
        except (OSError, urllib.error.URLError) as exc:
            raise RuntimeError(
                f"No se pudo contactar con el servidor VLM en "
                f"{self.server_url} ({exc}). ¿Está lanzado "
                f"`qwen3vl_server`?"
            ) from exc

        return data["choices"][0]["message"]["content"].strip()
