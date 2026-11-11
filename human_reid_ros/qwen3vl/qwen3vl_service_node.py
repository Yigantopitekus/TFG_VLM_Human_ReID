#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from human_reid_ros.image_utils import imgmsg_to_rgb
from human_reid_ros.qwen3vl.qwen3vl_engine import (
    DEFAULT_CACHE_DIR,
    DEFAULT_MODEL_ID,
    Qwen3VLEngine,
)
from human_reid_ros.srv import VlmInference


class Qwen3VLServiceNode(Node):
    def __init__(self):
        super().__init__("qwen3vl_service_node")

        self.declare_parameter("model_id", DEFAULT_MODEL_ID)
        self.declare_parameter("cache_dir", DEFAULT_CACHE_DIR)
        self.declare_parameter("max_new_tokens", 512)
        self.declare_parameter("max_image_side", 1280)

        model_id = self.get_parameter("model_id").value
        cache_dir = self.get_parameter("cache_dir").value
        self.get_logger().info(
            f"Cargando {model_id} (pesos en '{cache_dir}'). "
            "La primera vez se descargan (~17 GB), puede tardar..."
        )
        self.engine = Qwen3VLEngine(
            model_id=model_id,
            cache_dir=cache_dir,
            max_new_tokens=self.get_parameter("max_new_tokens").value,
            max_image_side=self.get_parameter("max_image_side").value,
        )

        self.srv = self.create_service(
            VlmInference, "/vlm_inference", self.inference_callback
        )
        self.get_logger().info("Servidor VLM activo (servicio 'vlm_inference').")

    def inference_callback(self, request, response):
        if len(request.references) == 0 or len(request.scene) == 0:
            response.success = False
            response.response = (
                "La petición debe incluir al menos una imagen de referencia y una de escena "
                f"(recibidas: {len(request.references)} ref., {len(request.scene)} escena)."
            )
            return response
        try:
            refs = [imgmsg_to_rgb(m) for m in request.references]
            scene = [imgmsg_to_rgb(m) for m in request.scene]
            response.response = self.engine.infer(refs, scene, request.prompt)
            response.success = True
        except Exception as e:  # noqa: BLE001 - el servicio no debe morir por una petición
            self.get_logger().error(f"Error en inferencia: {e}")
            response.success = False
            response.response = f"Error en inferencia: {e}"
        return response


def main(args=None):
    rclpy.init(args=args)
    node = Qwen3VLServiceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()