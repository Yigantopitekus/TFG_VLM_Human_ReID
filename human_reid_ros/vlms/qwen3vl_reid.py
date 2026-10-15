"""Re-ID de personas con Qwen3-VL-8B (lógica independiente de ROS).

Dos backends con la misma interfaz:

* ``server`` (por defecto): el modelo vive en un proceso aparte
  (``qwen3vl_server``); aquí solo se envían prompts + imágenes por HTTP.
  No se importa torch ni se carga ningún peso.
* ``local``: el modelo se carga dentro de este proceso, siempre desde
  disco (nunca se descarga nada).
"""
from PIL import Image

from .common import (
    DEFAULT_MAX_PIXELS,
    DEFAULT_MODEL_PATH,
    DEFAULT_SERVER_URL,
    limit_pixels,
)
from .prompts import QUERY_PROMPT, REFERENCES_PROMPT


class Qwen3VLReID:

    def __init__(self, backend="server", server_url=DEFAULT_SERVER_URL,
                 model_path=DEFAULT_MODEL_PATH, quantization="4bit",
                 timeout=300.0, max_pixels=DEFAULT_MAX_PIXELS):

        self.max_pixels = max_pixels
        self.chat_history = []
        self.remote = None

        if backend == "server":
            from .qwen3vl_remote import RemoteChat
            self.remote = RemoteChat(server_url, timeout=timeout)
            self._chat = self.remote.chat

        elif backend == "local":
            # Import diferido: solo aquí hace falta torch.
            from .qwen3vl_engine import Qwen3VLEngine
            engine = Qwen3VLEngine(model_path, quantization, max_pixels)
            self._chat = engine.chat

        else:
            raise ValueError(
                f"backend debe ser 'server' o 'local', no {backend!r}"
            )

    def _images_to_content(self, images):
        content = []
        for image in images:
            # OpenCV entrega BGR; PIL espera RGB.
            pil = Image.fromarray(image[:, :, ::-1])
            content.append({
                "type": "image",
                "image": limit_pixels(pil, self.max_pixels),
            })
        return content

    def describe_references(self, references):
        self.chat_history = []

        content = self._images_to_content(references)
        content.append({"type": "text", "text": REFERENCES_PROMPT})

        self.chat_history.append({"role": "user", "content": content})
        raw = self._chat(self.chat_history, max_new_tokens=512)
        self.chat_history.append({
            "role": "assistant",
            "content": [{"type": "text", "text": raw}],
        })

        return raw

    def describe_query(self, query_image):
        if not self.chat_history:
            raise RuntimeError(
                "Hay que llamar a describe_references() antes de describe_query()."
            )

        content = self._images_to_content([query_image])
        content.append({"type": "text", "text": QUERY_PROMPT})

        self.chat_history.append({"role": "user", "content": content})
        return self._chat(self.chat_history, max_new_tokens=512)
