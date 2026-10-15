import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from tu_paquete_ros.srv import VlmInference
from human_reid_ros.vlms.prompts import REID_PROMPT

class Qwen3VLReIDNode(Node):
    def __init__(self):
        super().__init__('qwen3vl_reid_node')
        self.vlm_client = self.create_client(VlmInference, 'vlm_inference')
        
        while not self.vlm_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Esperando servicio VLM...')
            
        self.subscription = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.is_processing = False

    def image_callback(self, msg):
        if self.is_processing:
            return
            
        self.is_processing = True
        req = VlmInference.Request()
        req.image = msg
        req.prompt = REID_PROMPT
        
        future = self.vlm_client.call_async(req)
        future.add_done_callback(self.response_callback)

    def response_callback(self, future):
        try:
            self.get_logger().info(f"ReID: {future.result().response}")
        except Exception as e:
            self.get_logger().error(f"Fallo: {e}")
        finally:
            self.is_processing = False

def main(args=None):
    rclpy.init(args=args)
    node = Qwen3VLReIDNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()