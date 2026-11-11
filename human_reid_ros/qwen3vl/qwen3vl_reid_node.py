#!/usr/bin/env python3
"""Nodo cliente ReID.

Protocolo con image_publisher (viaja en header.frame_id, sin interfaces nuevas):
    frame_id = "<nombre>#<lote>#<n>/<archivo>"
      - referencias: <nombre> = paciente      (ej. "paciente1#1730000000#2/paciente1_cam1.jpg")
      - escena:      <nombre> = "scene"       (ej. "scene#1730000100#2/frame_s33__image_raw.png")
      - <lote>: id del envío; un lote nuevo descarta el anterior (evita mezclar pacientes).
      - <n>: nº de imágenes del lote; para la escena indica cuándo está completa.
Si el frame_id no sigue este formato (otra fuente de imágenes), se admite igualmente:
todo cuenta como un único lote y cada imagen de escena dispara una inferencia.
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

from human_reid_ros.srv import VlmInference


def build_prompt(n_ref: int, n_scene: int) -> str:
    refs = f"Picture 1 a Picture {n_ref}" if n_ref > 1 else "Picture 1"
    first_scene = n_ref + 1
    last_scene = n_ref + n_scene
    escena = (
        f"Picture {first_scene} a Picture {last_scene}"
        if n_scene > 1
        else f"Picture {first_scene}"
    )
    return (
        f"Te paso {n_ref + n_scene} imágenes, cada una etiquetada como 'Picture N'.\n"
        f"- {refs}: REFERENCIA. Son recortes de UNA misma persona (la persona objetivo) "
        "tomados desde cámaras con ángulos distintos. Cada recorte es solo esa persona, "
        "sin el resto de la sala.\n"
        f"- {escena}: ESCENA. Son vistas de la sala completa tomadas desde las mismas cámaras "
        "en instantes de tiempo distintos. La persona objetivo puede aparecer en una, "
        "en varias o en ninguna. Las imágenes de REFERENCIA no forman parte de la escena.\n\n"
        "Sigue estos pasos de forma concisa (máximo 4 líneas por paso):\n"
        "1. Describe la ropa y los rasgos físicos de la persona objetivo (imágenes de referencia).\n"
        "2. Busca a la persona objetivo en las imágenes de la escena, indicando en cuáles aparece.\n"
        "3. Termina con una última línea exactamente así: 'VEREDICTO: SÍ' o 'VEREDICTO: NO', "
        "seguida de una justificación breve y de dónde está la persona "
        "('Picture N' y posición dentro de la imagen)."
    )


def parse_frame_id(frame_id: str):
    """-> (grupo, nombre, n | None). 'grupo' identifica el lote completo."""
    group = frame_id.rsplit("/", 1)[0]
    parts = group.split("#")
    n = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else None
    return group, parts[0], n


class Qwen3VLReIDNode(Node):
    def __init__(self):
        super().__init__("qwen3vl_reid_node")

        self.declare_parameter("reference_topic", "/human_reid/reference")
        self.declare_parameter("query_topic", "/human_reid/crowd")
        self.declare_parameter("max_references", 2)
        # Solo se usa si el publicador no indica cuántas imágenes tiene la escena
        self.declare_parameter("default_scene_images", 1)

        self.max_references = self.get_parameter("max_references").value
        self.default_scene_images = self.get_parameter("default_scene_images").value

        self.references = []        # Image de la persona actual
        self.ref_group = None
        self.patient = None
        self.scene = []             # Image de la escena en curso
        self.scene_group = None
        self.is_processing = False

        self.vlm_client = self.create_client(VlmInference, "/vlm_inference")

        self.get_logger().info("Buscando el servicio '/vlm_inference'...")
        while not self.vlm_client.wait_for_service(timeout_sec=2.0):
            if not rclpy.ok():
                self.get_logger().error("Interrumpido mientras se esperaba el servicio.")
                return
            self.get_logger().info("Servicio no disponible, esperando (¿sigue cargando el modelo?)...")
        self.get_logger().info("¡Servicio enlazado correctamente!")

        # Depth 10 en ambos: la escena llega como varias imágenes seguidas y con depth=1
        # se perdería alguna si el ejecutor aún no ha despachado la anterior.
        self.create_subscription(
            Image, self.get_parameter("reference_topic").value, self.reference_callback, 10
        )
        self.create_subscription(
            Image, self.get_parameter("query_topic").value, self.query_callback, 10
        )
        self.get_logger().info("Nodo ReID listo. Esperando referencias y escena.")

    def reference_callback(self, msg):
        group, name, _ = parse_frame_id(msg.header.frame_id)
        if group != self.ref_group:
            if self.references:
                self.get_logger().info(
                    f"Nuevo lote de referencias ('{name}'): descarto las {len(self.references)} anteriores."
                )
            self.references = []
            self.ref_group = group
            self.patient = name
        self.references.append(msg)
        if len(self.references) > self.max_references:
            self.references.pop(0)
        self.get_logger().info(f"Referencias de '{self.patient}': {len(self.references)}")

    def query_callback(self, msg):
        if self.is_processing:
            self.get_logger().warn("Inferencia en curso, imagen de escena descartada.")
            return

        group, _, n = parse_frame_id(msg.header.frame_id)
        if group != self.scene_group:
            self.scene = []
            self.scene_group = group
        self.scene.append(msg)

        expected = n or self.default_scene_images
        if len(self.scene) < expected:
            self.get_logger().info(f"Escena: {len(self.scene)}/{expected} imágenes recibidas.")
            return

        if not self.references:
            self.get_logger().warn("No hay imágenes de referencia todavía (usa 'ref <paciente>').")
            self.scene = []
            return
        if not self.vlm_client.service_is_ready():
            self.get_logger().warn("El servicio 'vlm_inference' no está disponible.")
            self.scene = []
            return

        refs, scene = list(self.references), list(self.scene)
        self.scene = []
        self.is_processing = True
        self.get_logger().info(
            f"Consultando al VLM: paciente '{self.patient}' ({len(refs)} ref.) + escena ({len(scene)} img.)..."
        )
        req = VlmInference.Request()
        req.references = refs
        req.scene = scene
        req.prompt = build_prompt(len(refs), len(scene))
        patient = self.patient
        self.vlm_client.call_async(req).add_done_callback(
            lambda fut, p=patient: self.response_callback(fut, p)
        )

    def response_callback(self, future, patient=""):
        try:
            result = future.result()
            if result.success:
                self.get_logger().info(f"ReID [{patient}]:\n{result.response}")
            else:
                self.get_logger().error(f"El servidor devolvió error: {result.response}")
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f"Fallo en la llamada al servicio: {e}")
        finally:
            self.is_processing = False


def main(args=None):
    rclpy.init(args=args)
    node = Qwen3VLReIDNode()
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