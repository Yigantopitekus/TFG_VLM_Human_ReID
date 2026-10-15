import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from tu_paquete_ros.srv import VlmInference 
from human_reid_ros.vlms.qwen3vl_engine import Qwen3VLEngine

class Qwen3VLServiceNode(Node):
    def __init__(self):
        super().__init__('qwen3vl_service_node')
        self.get_logger().info("Cargando modelo Qwen3-VL en VRAM...")
        
        self.engine = Qwen3VLEngine()
        self.bridge = CvBridge()
        self.srv = self.create_service(VlmInference, 'vlm_inference', self.inference_callback)
        
        self.get_logger().info("Servidor VLM activo.")

    def inference_callback(self, request, response):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(request.image, desired_encoding='passthrough')
            response.response = self.engine.process_image(cv_image, request.prompt)
        except Exception as e:
            self.get_logger().error(f"Error: {e}")
            response.response = "Error en inferencia."
        return response

def main(args=None):
    rclpy.init(args=args)
    node = Qwen3VLServiceNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()