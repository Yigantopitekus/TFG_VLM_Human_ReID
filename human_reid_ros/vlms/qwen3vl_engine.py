"""Motor de inferencia de Qwen3-VL-8B (transformers).

Es el único módulo que importa torch/transformers. Lo usan:
  * el servidor independiente (``qwen3vl_server``), y
  * el modo ``backend:=local`` del nodo ROS.

NUNCA descarga pesos: ``local_files_only=True`` + modo offline de Hugging
Face. Los pesos se instalan una sola vez con ``qwen3vl_download``.
"""
import os

# Debe fijarse antes de importar transformers / huggingface_hub.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402
from transformers import (  # noqa: E402
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen3VLForConditionalGeneration,
)

from .common import DEFAULT_MAX_PIXELS, DEFAULT_MODEL_PATH, limit_pixels  # noqa: E402

QUANTIZATION_MODES = ("4bit", "8bit", "none")


def _compute_dtype():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


class Qwen3VLEngine:

    def __init__(self, model_path=DEFAULT_MODEL_PATH, quantization="4bit",
                 max_pixels=DEFAULT_MAX_PIXELS):

        if quantization not in QUANTIZATION_MODES:
            raise ValueError(
                f"quantization debe ser una de {QUANTIZATION_MODES}, "
                f"no {quantization!r}"
            )

        self.model_path = os.path.expanduser(model_path)
        self.max_pixels = max_pixels
        dtype = _compute_dtype()

        kwargs = dict(device_map="auto", local_files_only=True)

        if quantization == "4bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
                # El codificador visual se deja en precisión completa: cuantizarlo
                # empeora la percepción de colores/detalles de la ropa.
                llm_int8_skip_modules=["visual"],
            )
        elif quantization == "8bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_skip_modules=["visual"],
            )
        else:
            kwargs["dtype"] = dtype

        print(f"Cargando Qwen3-VL-8B desde '{self.model_path}' "
              f"({quantization}, sin descargas)...")

        try:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_path, **kwargs
            )
            self.processor = AutoProcessor.from_pretrained(
                self.model_path, local_files_only=True
            )
        except OSError as exc:
            raise RuntimeError(
                f"No se encontraron los pesos en '{self.model_path}'. "
                "Instálalos una vez con:  qwen3vl_download  "
                "(o: python -m human_reid_ros.qwen3vl_download)"
            ) from exc

        self.model.eval()
        print("Qwen3-VL-8B cargado.")

    def chat(self, messages, max_new_tokens=512):
        """``messages`` en formato interno (imágenes PIL). Devuelve el texto."""

        images = []
        template_messages = []

        for message in messages:
            parts = []
            for part in message["content"]:
                if part["type"] == "image":
                    images.append(limit_pixels(part["image"], self.max_pixels))
                    parts.append({"type": "image"})
                else:
                    parts.append({"type": "text", "text": part["text"]})
            template_messages.append(
                {"role": message["role"], "content": parts}
            )

        text = self.processor.apply_chat_template(
            template_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.processor(
            text=[text],
            images=images or None,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
            )

        trimmed = generated_ids[:, inputs["input_ids"].shape[1]:]

        output = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return output[0].strip()
