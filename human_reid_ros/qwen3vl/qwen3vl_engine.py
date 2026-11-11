import os

import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig

DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_CACHE_DIR = "/opt/data"


class Qwen3VLEngine:
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        cache_dir: str = DEFAULT_CACHE_DIR,
        max_new_tokens: int = 512,
        max_image_side: int = 1280,
    ):
        cache_dir = os.path.expanduser(cache_dir)
        os.makedirs(cache_dir, exist_ok=True)

        self.max_new_tokens = max_new_tokens
        self.max_image_side = max_image_side

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        if "Qwen2.5" in model_id:
            from transformers import Qwen2_5_VLForConditionalGeneration
            ModelClass = Qwen2_5_VLForConditionalGeneration
        else:
            from transformers import Qwen3VLForConditionalGeneration
            ModelClass = Qwen3VLForConditionalGeneration

        self.model = ModelClass.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            device_map="auto",
            quantization_config=quantization_config,
        )
        self.processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)

    def infer(self, ref_images: list, scene_images: list, prompt: str) -> str:
        """
        ref_images / scene_images: listas de arrays numpy uint8 HxWx3 en RGB.
        Orden fijo: referencias primero, escena después.
        """
        content = []
        for arr in list(ref_images) + list(scene_images):
            content.append({
                "type": "image",
                "image": Image.fromarray(arr),
                "max_pixels": self.max_image_side * self.max_image_side,
            })
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            add_vision_id=True,  # etiqueta cada imagen como "Picture 1:", "Picture 2:"...
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)

        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]