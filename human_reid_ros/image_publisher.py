import os
import time

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2
from PIL import Image as PILImage, ImageOps
import numpy as np


class ImagePublisher(Node):

    def __init__(self):
        super().__init__("image_publisher")

        self.bridge = CvBridge()

        self.reference_publisher = self.create_publisher(
            Image,
            "/human_reid/reference",
            10
        )

        self.crowd_publisher = self.create_publisher(
            Image,
            "/human_reid/crowd",
            10
        )

        # Carpeta de referencias e imagen de crowd (configurables con
        # --ros-args -p reference_dir:=... -p crowd_image_path:=...)
        self.declare_parameter(
            "reference_dir",
            "~/TFG/vlm_human_reid/images/references"
        )
        self.declare_parameter(
            "crowd_image_path",
            "~/TFG/vlm_human_reid/images/crowd/crowd4.jpg"
        )

        self.reference_dir = os.path.expanduser(
            self.get_parameter("reference_dir").value
        )
        self.crowd_image_path = os.path.expanduser(
            self.get_parameter("crowd_image_path").value
        )

        self.get_logger().info("Image Publisher iniciado.")
        self.get_logger().info(
            "Comandos disponibles: ref | crowd | exit"
        )

    def load_image_exif_corrected(self, path):
        pil_img = PILImage.open(path)
        pil_img = ImageOps.exif_transpose(pil_img)  # no-op si no hay tag EXIF (p.ej. PNG)
        pil_img = pil_img.convert("RGB")
        rgb_array = np.array(pil_img)
        return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

    def wait_for_reference_subscriber(self):
        self.get_logger().info("Esperando al nodo VLM...")
        while (
            self.reference_publisher.get_subscription_count() == 0
            and rclpy.ok()
        ):
            time.sleep(0.1)
        self.get_logger().info("Nodo VLM conectado.")

    def wait_for_crowd_subscriber(self):
        self.get_logger().info("Esperando al nodo VLM...")
        while (
            self.crowd_publisher.get_subscription_count() == 0
            and rclpy.ok()
        ):
            time.sleep(0.1)
        self.get_logger().info("Nodo VLM conectado.")

    def publish_references(self):

        if not os.path.isdir(self.reference_dir):
            self.get_logger().error(
                f"No existe la carpeta: {self.reference_dir}"
            )
            return

        image_files = sorted([
            file
            for file in os.listdir(self.reference_dir)
            if file.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        if len(image_files) == 0:
            self.get_logger().error("No se encontraron imágenes de referencia.")
            return

        self.get_logger().info(f"Encontradas {len(image_files)} referencias.")

        self.wait_for_reference_subscriber()

        for image_file in image_files:

            image_path = os.path.join(self.reference_dir, image_file)

            try:
                image = self.load_image_exif_corrected(image_path)
            except Exception as e:
                self.get_logger().warn(f"No se pudo leer: {image_path} ({e})")
                continue

            msg = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")
            self.reference_publisher.publish(msg)
            self.get_logger().info(f"Referencia publicada: {image_file}")

            time.sleep(0.5)

        self.get_logger().info("Todas las referencias han sido publicadas.")

    def publish_crowd(self):

        if not os.path.isfile(self.crowd_image_path):
            self.get_logger().error(f"No existe la imagen: {self.crowd_image_path}")
            return

        try:
            image = self.load_image_exif_corrected(self.crowd_image_path)
        except Exception as e:
            self.get_logger().error(f"No se pudo leer la imagen de crowd: {e}")
            return

        self.wait_for_crowd_subscriber()

        msg = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")
        self.crowd_publisher.publish(msg)
        self.get_logger().info("Imagen de crowd publicada.")


def main(args=None):

    rclpy.init(args=args)
    node = ImagePublisher()

    try:
        while rclpy.ok():
            command = input(
                "\nEscribe un comando (ref / crowd / exit): "
            ).strip().lower()

            if command == "ref":
                node.get_logger().info("Publicando referencias...")
                node.publish_references()

            elif command == "crowd":
                node.get_logger().info("Publicando imagen de crowd...")
                node.publish_crowd()

            elif command == "exit":
                node.get_logger().info("Cerrando Image Publisher.")
                break

            else:
                node.get_logger().warn("Comando no reconocido. Usa: ref, crowd o exit.")

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()