"""Constantes y utilidades compartidas por el cliente, el motor y el servidor.

Este módulo NO importa torch ni ROS, de modo que el nodo ROS en modo
``server`` no necesita ninguna dependencia de deep learning.

Formato interno de mensajes (el mismo que usa transformers):

    {"role": "user", "content": [
        {"type": "image", "image": <PIL.Image>},
        {"type": "text",  "text": "..."},
    ]}

Formato de transporte (API tipo OpenAI ``/v1/chat/completions``):

    {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
        {"type": "text", "text": "..."},
    ]}
"""
import base64
import io
import math

from PIL import Image

# Único modelo soportado.
REPO_ID = "Qwen/Qwen3-VL-8B-Instruct"

# Carpeta donde `qwen3vl_download` instala los pesos (descarga única).
DEFAULT_MODEL_PATH = "~/TFG/vlm_human_reid/models/Qwen3-VL-8B-Instruct"

DEFAULT_SERVER_URL = "http://127.0.0.1:8000"

# Qwen3-VL usa parches de 16 px fusionados 2x2 -> 32 px por token visual.
# 384 tokens visuales por imagen (equivalente al 384*28*28 de Qwen2.5-VL).
DEFAULT_MAX_PIXELS = 384 * 32 * 32


def limit_pixels(image, max_pixels=DEFAULT_MAX_PIXELS):
    """Reduce la imagen (manteniendo proporción) si supera ``max_pixels``."""
    width, height = image.size
    if width * height <= max_pixels:
        return image
    scale = math.sqrt(max_pixels / (width * height))
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return image.resize(new_size, Image.LANCZOS)


def pil_to_data_uri(image, quality=95):
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return "data:image/jpeg;base64," + encoded


def data_uri_to_pil(uri):
    if not uri.startswith("data:"):
        raise ValueError(
            "Solo se admiten imágenes incrustadas (data URI en base64); "
            "el servidor no descarga nada de internet."
        )
    try:
        encoded = uri.split(",", 1)[1]
        raw = base64.b64decode(encoded)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Imagen base64 inválida: {exc}") from exc


def to_openai_messages(messages):
    """Formato interno (PIL) -> formato de transporte (data URI)."""
    out = []
    for message in messages:
        parts = []
        for part in message["content"]:
            if part["type"] == "image":
                parts.append({
                    "type": "image_url",
                    "image_url": {"url": pil_to_data_uri(part["image"])},
                })
            else:
                parts.append({"type": "text", "text": part["text"]})

        # Los mensajes solo-texto (p. ej. el assistant) se envían como str,
        # que es lo que aceptan todos los servidores compatibles.
        if all(p["type"] == "text" for p in parts):
            content = "\n".join(p["text"] for p in parts)
        else:
            content = parts
        out.append({"role": message["role"], "content": content})
    return out


def from_openai_messages(messages):
    """Formato de transporte (data URI) -> formato interno (PIL)."""
    out = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            parts = [{"type": "text", "text": content}]
        elif isinstance(content, list):
            parts = []
            for part in content:
                kind = part.get("type")
                if kind == "text":
                    parts.append({"type": "text", "text": part["text"]})
                elif kind == "image_url":
                    url = part["image_url"]
                    url = url["url"] if isinstance(url, dict) else url
                    parts.append({"type": "image", "image": data_uri_to_pil(url)})
                else:
                    raise ValueError(f"Tipo de contenido no soportado: {kind!r}")
        else:
            raise ValueError("Cada mensaje necesita 'content' (str o lista).")
        out.append({"role": message["role"], "content": parts})
    return out
