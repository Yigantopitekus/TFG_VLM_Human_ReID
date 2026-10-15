import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

from .vlms.common import DEFAULT_MODEL_PATH, DEFAULT_SERVER_URL
from .vlms.qwen3vl_reid import Qwen3VLReID

NUM_REFERENCES = 5


class Qwen3VLReIDNode(Node):

    def __init__(self):

        super().__init__("qwen3vl_reid_node")

        # backend = "server": el modelo corre en un proceso aparte
        #                     (qwen3vl_server); este nodo no carga pesos.
        # backend = "local" : el modelo se carga en este proceso (desde disco).
        self.declare_parameter("backend", "server")
        self.declare_parameter("server_url", DEFAULT_SERVER_URL)
        self.declare_parameter("request_timeout", 300.0)
        self.declare_parameter("model_path", DEFAULT_MODEL_PATH)  # solo local
        self.declare_parameter("quantization", "4bit")            # solo local

        backend = self.get_parameter("backend").value

        self.bridge = CvBridge()
        self.references = []
        self.reference_profile = None

        self.reference_subscription = self.create_subscription(
            Image,
            "/human_reid/reference",
            self.reference_callback,
            10
        )

        self.crowd_subscription = self.create_subscription(
            Image,
            "/human_reid/crowd",
            self.crowd_callback,
            10
        )

        self.result_publisher = self.create_publisher(
            String,
            "/human_reid/result",
            10
        )

        if backend == "server":
            server_url = self.get_parameter("server_url").value
            self.get_logger().info(
                f"Qwen3-VL-8B en servidor independiente: {server_url}"
            )
        else:
            self.get_logger().info("Cargando Qwen3-VL-8B en este proceso...")

        self.qwen = Qwen3VLReID(
            backend=backend,
            server_url=self.get_parameter("server_url").value,
            model_path=self.get_parameter("model_path").value,
            quantization=self.get_parameter("quantization").value,
            timeout=self.get_parameter("request_timeout").value,
        )

        if self.qwen.remote is not None:
            if self.qwen.remote.is_ready():
                self.get_logger().info("Servidor VLM conectado.")
            else:
                self.get_logger().warn(
                    "El servidor VLM no responde todavía. Lánzalo con "
                    "`python -m human_reid_ros.qwen3vl_server`; las "
                    "peticiones fallarán hasta que esté listo."
                )

    def reference_callback(self, msg):

        reference = self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding="bgr8"
        )

        self.references.append(reference)

        self.get_logger().info(
            f"Imagen de referencia recibida "
            f"({len(self.references)}/{NUM_REFERENCES} referencias)"
        )

        # Generamos el perfil una sola vez al recibir las referencias.
        if len(self.references) == NUM_REFERENCES:

            self.get_logger().info(
                "Generando perfil de identidad con Qwen3-VL-8B..."
            )

            try:
                self.reference_profile = self.qwen.describe_references(
                    self.references
                )

                print("\n" + "=" * 60)
                print("PERFIL DE REFERENCIA")
                print("=" * 60)
                print(self.reference_profile)
                print("=" * 60)

            except Exception as e:
                self.get_logger().error(
                    f"Error generando perfil: {e}"
                )

    def crowd_callback(self, msg):

        if self.reference_profile is None:
            self.get_logger().warn(
                "No hay perfil de referencia activo. Ignorando query."
            )
            return

        self.get_logger().info(
            "Imagen query recibida. Comparando con perfil..."
        )

        query = self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding="bgr8"
        )

        try:
            query_result = self.qwen.describe_query(query)

            print("\n" + "=" * 60)
            print("RESULTADO QUERY (MATCH / NO MATCH)")
            print("=" * 60)
            print(query_result)
            print("=" * 60)

            msg_out = String()
            msg_out.data = query_result
            self.result_publisher.publish(msg_out)

        except Exception as e:
            self.get_logger().error(
                f"Error procesando query: {e}"
            )

        # Reiniciamos tras procesar la query para empezar un ciclo nuevo.
        self.reset()

    def reset(self):
        self.references.clear()
        self.reference_profile = None

        self.get_logger().info(
            "Referencias y perfil borrados para el siguiente ciclo."
        )


def main(args=None):

    rclpy.init(args=args)

    node = Qwen3VLReIDNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
