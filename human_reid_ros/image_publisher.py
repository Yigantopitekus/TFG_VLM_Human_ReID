#!/usr/bin/env python3
import os
import shlex
import time

import numpy as np
import rclpy
from PIL import Image as PILImage
from PIL import ImageOps
from rclpy.node import Node
from sensor_msgs.msg import Image

from human_reid_ros.image_utils import rgb_to_imgmsg

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


class ImagePublisher(Node):
    def __init__(self):
        super().__init__("image_publisher")

        self.reference_publisher = self.create_publisher(Image, "/human_reid/reference", 10)
        self.crowd_publisher = self.create_publisher(Image, "/human_reid/crowd", 10)

        # Carpeta base con una subcarpeta por paciente (saved_frames/paciente1, ...)
        self.declare_parameter("reference_dir", "/home/alumnos/tito/TFG/TFG_VLM_Human_ReID/saved_frames")
        # Carpeta con las imágenes de la escena (o una sola imagen)
        self.declare_parameter("crowd_image_path", "/home/alumnos/tito/TFG/TFG_VLM_Human_ReID/scene")
        self.declare_parameter("subscriber_timeout_sec", 15.0)

        self.reference_dir = os.path.expanduser(self.get_parameter("reference_dir").value)
        self.crowd_image_path = os.path.expanduser(self.get_parameter("crowd_image_path").value)
        self.subscriber_timeout = self.get_parameter("subscriber_timeout_sec").value

        self.get_logger().info(
            "Image Publisher iniciado. Comandos: list | ref <paciente> [fotos...] | crowd [fotos...] | exit"
        )

    def load_rgb(self, path) -> np.ndarray:
        img = PILImage.open(path)
        img = ImageOps.exif_transpose(img)
        return np.array(img.convert("RGB"))

    def wait_for_subscriber(self, publisher) -> bool:
        deadline = time.monotonic() + self.subscriber_timeout
        while publisher.get_subscription_count() == 0 and rclpy.ok():
            if time.monotonic() > deadline:
                self.get_logger().error(
                    f"Ningún suscriptor en {publisher.topic_name} tras "
                    f"{self.subscriber_timeout:.0f}s. ¿Está lanzado qwen3vl_reid_node "
                    "(solo se suscribe cuando el servicio del modelo ya está activo)?"
                )
                return False
            time.sleep(0.1)
        return rclpy.ok()

    @staticmethod
    def list_images(folder):
        return sorted(f for f in os.listdir(folder) if f.lower().endswith(IMAGE_EXTENSIONS))

    @staticmethod
    def match_files(available, requested):
        """Resuelve nombres pedidos (con o sin extensión) contra los disponibles."""
        selected, missing = [], []
        for req in requested:
            match = next(
                (f for f in available if f == req or os.path.splitext(f)[0] == req), None
            )
            if match is None:
                missing.append(req)
            elif match not in selected:
                selected.append(match)
        return selected, missing

    def make_group(self, name, count):
        """Identificador de lote que viaja en frame_id: '<nombre>#<lote>#<n>'."""
        return f"{name}#{time.time_ns()}#{count}"

    def list_patients(self):
        if not os.path.isdir(self.reference_dir):
            self.get_logger().error(f"No existe reference_dir: {self.reference_dir}")
            return
        patients = sorted(
            d for d in os.listdir(self.reference_dir)
            if os.path.isdir(os.path.join(self.reference_dir, d))
        )
        for p in patients:
            n = len(self.list_images(os.path.join(self.reference_dir, p)))
            self.get_logger().info(f"  {p} ({n} imágenes)")
        if not patients:
            self.get_logger().warn("No hay carpetas de pacientes.")

    def publish_references(self, patient_name: str, requested_files=None):
        patient_dir = os.path.join(self.reference_dir, patient_name)

        if not os.path.isdir(patient_dir):
            self.get_logger().error(f"No existe la carpeta del paciente: {patient_dir}")
            return

        available = self.list_images(patient_dir)
        if not available:
            self.get_logger().error(f"No se encontraron imágenes de referencia en {patient_dir}.")
            return

        if requested_files:
            files, missing = self.match_files(available, requested_files)
            if missing:
                self.get_logger().warn(
                    f"No encontradas para {patient_name}: {missing}. Disponibles: {available}"
                )
            if not files:
                return
        else:
            files = available

        # Cargamos todo antes de publicar: si algo falla, no se envía un lote a medias
        loaded = []
        for name in files:
            path = os.path.join(patient_dir, name)
            try:
                loaded.append((name, self.load_rgb(path)))
            except Exception as e:  # noqa: BLE001
                self.get_logger().warn(f"No se pudo leer: {path} ({e})")
        if not loaded:
            return

        if not self.wait_for_subscriber(self.reference_publisher):
            return

        group = self.make_group(patient_name, len(loaded))
        self.get_logger().info(f"Publicando {len(loaded)} referencia(s) de {patient_name}.")
        for name, rgb in loaded:
            self.reference_publisher.publish(
                rgb_to_imgmsg(rgb, self.get_clock().now().to_msg(), f"{group}/{name}")
            )
            self.get_logger().info(f"Referencia de {patient_name} publicada: {name}")
            time.sleep(0.5)

        self.get_logger().info(f"Referencias de {patient_name} publicadas.")

    def resolve_crowd_files(self, requested=None):
        """Devuelve la lista de rutas de escena a publicar (todas las de la carpeta por defecto)."""
        path = self.crowd_image_path
        if os.path.isfile(path):
            return [path]
        if not os.path.isdir(path):
            self.get_logger().error(f"No existe: {path}")
            return []
        available = self.list_images(path)
        if not available:
            self.get_logger().error(f"No hay imágenes en {path}")
            return []
        if requested:
            files, missing = self.match_files(available, requested)
            if missing:
                self.get_logger().warn(f"No encontradas en la escena: {missing}. Disponibles: {available}")
        else:
            files = available
        return [os.path.join(path, f) for f in files]

    def publish_crowd(self, requested=None):
        paths = self.resolve_crowd_files(requested)
        if not paths:
            return

        loaded = []
        for path in paths:
            try:
                loaded.append((os.path.basename(path), self.load_rgb(path)))
            except Exception as e:  # noqa: BLE001
                self.get_logger().error(f"No se pudo leer la imagen de escena {path}: {e}")
                return  # lote incompleto: mejor no enviar nada

        if not self.wait_for_subscriber(self.crowd_publisher):
            return

        group = self.make_group("scene", len(loaded))
        for name, rgb in loaded:
            self.crowd_publisher.publish(
                rgb_to_imgmsg(rgb, self.get_clock().now().to_msg(), f"{group}/{name}")
            )
            self.get_logger().info(f"Imagen de escena publicada: {name}")
            time.sleep(0.2)
        self.get_logger().info(f"Escena publicada ({len(loaded)} imagen/es). Esperando respuesta del VLM...")


def main(args=None):
    rclpy.init(args=args)
    node = ImagePublisher()
    try:
        while rclpy.ok():
            raw = input("\nComando (list / ref <paciente> [fotos...] / crowd [fotos...] / exit): ").strip()
            try:
                parts = shlex.split(raw)  # sin .lower(): las rutas distinguen mayúsculas
            except ValueError as e:
                node.get_logger().warn(f"Comando mal formado: {e}")
                continue
            if not parts:
                continue

            cmd, cmd_args = parts[0].lower(), parts[1:]

            if cmd == "list":
                node.list_patients()
            elif cmd == "ref":
                if not cmd_args:
                    node.list_patients()
                    name = input("Paciente: ").strip()
                    if not name:
                        continue
                    cmd_args = [name]
                node.publish_references(cmd_args[0], cmd_args[1:] or None)
            elif cmd == "crowd":
                node.publish_crowd(cmd_args or None)
            elif cmd == "exit":
                break
            else:
                node.get_logger().warn("Comando no reconocido.")
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()