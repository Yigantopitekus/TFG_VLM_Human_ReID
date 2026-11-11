#!/usr/bin/env python3
import os
import re

import cv2
import message_filters
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from ultralytics import YOLO

# Colores BGR para depuración
COLOR_IN_ROI = (0, 255, 0)      # verde: persona cuyo centroide está dentro de la ROI
COLOR_OUT_ROI = (0, 0, 255)     # rojo: persona detectada fuera de la ROI
COLOR_ROI = (255, 128, 0)       # azul: borde de la ROI


class DualCameraPersonDetector(Node):
    def __init__(self):
        super().__init__('person_detector')

        # Parámetros
        self.declare_parameter('save_dir', 'saved_frames')
        # Topics donde se publica el vídeo con ROI y cajas (solo para visualizar, p. ej. rqt_image_view)
        self.declare_parameter('debug_topic_cam1', '/person_detector/cam1/debug')
        self.declare_parameter('debug_topic_cam2', '/person_detector/cam2/debug')
        # Margen extra alrededor de la caja al recortar, como fracción de su tamaño (0.0 = justo la caja)
        self.declare_parameter('crop_margin', 0.0)
        self.declare_parameter('conf_threshold', 0.5)    # confianza mínima (estricta: > umbral)
        self.declare_parameter('cam2_square_frac', 0.25)  # lado del cuadrado central de cam2, fracción de la altura
        # Si entre dos frames sincronizados pasan más de estos segundos (según header.stamp),
        # se considera que el stream se cortó y se olvida si había alguien en la ROI.
        self.declare_parameter('stream_gap_sec', 1.0)
        self.save_dir = self.get_parameter('save_dir').value
        self.debug_topic_cam1 = self.get_parameter('debug_topic_cam1').value
        self.debug_topic_cam2 = self.get_parameter('debug_topic_cam2').value
        self.crop_margin = self.get_parameter('crop_margin').value
        self.conf_threshold = self.get_parameter('conf_threshold').value
        self.cam2_square_frac = self.get_parameter('cam2_square_frac').value
        self.stream_gap_sec = self.get_parameter('stream_gap_sec').value

        self.bridge = CvBridge()
        self.person_was_present = False
        self.last_frame_stamp = None  # header.stamp (s) del último par sincronizado procesado
        self.last_save_time = self.get_clock().now()
        self.cooldown_seconds = 1.5
        # Cargar el modelo YOLOv8 Nano
        self.yolo = YOLO('yolov8n.pt')

        # Crear directorio para guardar las imágenes
        os.makedirs(self.save_dir, exist_ok=True)
        # Continúa la numeración tras reiniciar el nodo (no sobrescribe pacientes anteriores)
        self.frame_count = self.last_patient_index(self.save_dir)

        # Zonas de interés [x_min, y_min, x_max, y_max] en porcentajes (0.0 a 1.0)
        # cam1 (OAK): cuarto derecho de la imagen (x de 0.75 a 1.0), de arriba a abajo.
        self.roi_cam1 = [2.1 / 3.0, 0, 1.0, 1.0]
        # cam2: cuadrado centrado; depende de la resolución, se calcula por frame
        # (ver centered_square_roi).

        # Publicadores de las imágenes anotadas (cajas + ROI). QoS por defecto (reliable)
        # para que rqt_image_view / rviz conecten sin configurar nada.
        self.debug_pub1 = self.create_publisher(Image, self.debug_topic_cam1, 10)
        self.debug_pub2 = self.create_publisher(Image, self.debug_topic_cam2, 10)

        # Suscriptores a las dos cámaras
        sub_cam1 = message_filters.Subscriber(self, Image, '/oak/rgb/image_raw')
        sub_cam2 = message_filters.Subscriber(self, Image, '/image_raw')

        # Sincronizador de tiempo (aproximado por si hay ligero lag)
        # queue_size = 10, slop = 0.1 segundos de diferencia máxima
        self.ts = message_filters.ApproximateTimeSynchronizer([sub_cam1, sub_cam2], 10, 0.1)
        self.ts.registerCallback(self.sync_callback)

        self.get_logger().info("Nodo inicializado. Esperando frames sincronizados...")

    @staticmethod
    def last_patient_index(save_dir):
        """Mayor N entre las carpetas 'pacienteN' existentes en save_dir (0 si no hay ninguna)."""
        indices = [0]
        for name in os.listdir(save_dir):
            m = re.fullmatch(r'paciente(\d+)', name)
            if m and os.path.isdir(os.path.join(save_dir, name)):
                indices.append(int(m.group(1)))
        return max(indices)

    @staticmethod
    def centered_square_roi(img_width, img_height, side_frac_of_height):
        """
        ROI cuadrada (en píxeles) centrada en la imagen, con lado = side_frac_of_height * alto.
        Se devuelve en porcentajes [x_min, y_min, x_max, y_max] como el resto de ROIs.
        """
        side_px = side_frac_of_height * img_height
        half_w = (side_px / 2.0) / img_width
        half_h = side_frac_of_height / 2.0
        return [0.5 - half_w, 0.5 - half_h, 0.5 + half_w, 0.5 + half_h]

    @staticmethod
    def pick_target(detections):
        """Persona a guardar: la de mayor confianza entre las que están en la ROI (None si no hay)."""
        in_roi = [d for d in detections if d[5]]
        return max(in_roi, key=lambda d: d[4]) if in_roi else None

    @staticmethod
    def crop_detection(image, det, margin=0.0):
        """Recorte de la caja de det (con margen opcional), acotado a la imagen. None si queda vacío."""
        if det is None:
            return None
        h, w = image.shape[:2]
        x1, y1, x2, y2 = det[:4]
        mx, my = (x2 - x1) * margin, (y2 - y1) * margin
        x1, y1 = max(0, int(round(x1 - mx))), max(0, int(round(y1 - my)))
        x2, y2 = min(w, int(round(x2 + mx))), min(h, int(round(y2 + my)))
        if x2 - x1 < 2 or y2 - y1 < 2:
            return None
        return image[y1:y2, x1:x2].copy()

    def get_person_detections(self, results, roi, img_width, img_height):
        """
        Devuelve todas las personas detectadas como lista de tuplas
        (x1, y1, x2, y2, conf, in_roi), con coordenadas en píxeles.
        in_roi indica si el centroide de la caja cae dentro de la ROI.
        roi = [x_min, y_min, x_max, y_max] en porcentajes (0.0 a 1.0)
        """
        detections = []
        for box in results[0].boxes:
            if int(box.cls[0]) != 0:
                continue
            conf = float(box.conf[0])
            if conf <= self.conf_threshold:  
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            center_x = (x1 + x2) / 2.0 / img_width
            center_y = (y1 + y2) / 2.0 / img_height
            in_roi = (roi[0] <= center_x <= roi[2]) and (roi[1] <= center_y <= roi[3])
            detections.append((x1, y1, x2, y2, conf, in_roi))
        return detections

    def draw_debug(self, image, detections, roi, title):
        """
        Devuelve una COPIA de la imagen con la ROI y las cajas dibujadas.
        Verde y grueso: persona dentro de la ROI (la que dispara el guardado).
        Rojo y fino: persona fuera de la ROI.
        """
        out = image.copy()
        h, w = out.shape[:2]
        scale = max(0.5, w / 1280.0)                 
        thick_in = max(2, int(3 * scale))
        thick_out = max(1, int(1.5 * scale))
        font = cv2.FONT_HERSHEY_SIMPLEX

        rx1, ry1 = int(roi[0] * w), int(roi[1] * h)
        rx2, ry2 = int(roi[2] * w), int(roi[3] * h)
        cv2.rectangle(out, (rx1, ry1), (rx2, ry2), COLOR_ROI, thick_out + 1)
        cv2.putText(out, 'ROI', (rx1 + 5, ry2 - 8), font, 0.7 * scale, COLOR_ROI, thick_out + 1)

        for x1, y1, x2, y2, conf, in_roi in detections:
            color = COLOR_IN_ROI if in_roi else COLOR_OUT_ROI
            thickness = thick_in if in_roi else thick_out
            p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
            cv2.rectangle(out, p1, p2, color, thickness)

            # Centroide: es el punto que se compara con la ROI
            center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            cv2.circle(out, center, max(4, int(6 * scale)), color, -1)

            label = f"person {conf:.2f}" + (" IN ROI" if in_roi else "")
            (tw, th), _ = cv2.getTextSize(label, font, 0.6 * scale, thick_out)
            ty = p1[1] - 6 if p1[1] - th - 8 > 0 else p1[1] + th + 8  # dentro si no cabe arriba
            cv2.rectangle(out, (p1[0], ty - th - 4), (p1[0] + tw + 4, ty + 4), color, -1)
            cv2.putText(out, label, (p1[0] + 2, ty), font, 0.6 * scale, (0, 0, 0), thick_out)

        # Título de la cámara y nº de personas
        n_in = sum(1 for d in detections if d[5])
        header = f"{title} | personas: {len(detections)} | en ROI: {n_in}"
        hy = h - int(12 * scale)  # abajo, para no tapar las etiquetas de las cajas
        cv2.putText(out, header, (10, hy), font, 0.8 * scale, (0, 0, 0), thick_out + 3)
        cv2.putText(out, header, (10, hy), font, 0.8 * scale, (255, 255, 255), thick_out)
        return out

    def sync_callback(self, msg1, msg2):
        # Corte del stream (cámara/grabación interrumpida, o salto atrás al reiniciar un bag):
        # el estado 'había alguien en la ROI' ya no es fiable. Sin este reset, si al volver hay
        # OTRA persona en la ROI nunca se vería el flanco de subida y no se guardaría.
        stamp = msg1.header.stamp.sec + msg1.header.stamp.nanosec * 1e-9
        if self.last_frame_stamp is not None and abs(stamp - self.last_frame_stamp) > self.stream_gap_sec:
            if self.person_was_present:
                self.get_logger().warn(
                    f"Corte en el stream ({abs(stamp - self.last_frame_stamp):.2f} s): "
                    "reinicio el estado de presencia en la ROI.")
            self.person_was_present = False
        self.last_frame_stamp = stamp

        # Convertir mensajes ROS a imágenes OpenCV
        cv_image1 = self.bridge.imgmsg_to_cv2(msg1, desired_encoding='bgr8')
        cv_image2 = self.bridge.imgmsg_to_cv2(msg2, desired_encoding='bgr8')

        h1, w1, _ = cv_image1.shape
        h2, w2, _ = cv_image2.shape

        # ROI de cam2: cuadrado centrado de ~1/4 de la altura
        roi_cam2 = self.centered_square_roi(w2, h2, self.cam2_square_frac)

        # Inferencia con YOLO (verbose=False para no saturar la terminal)
        res1 = self.yolo.predict(cv_image1, classes=[0], conf=self.conf_threshold, verbose=False)
        res2 = self.yolo.predict(cv_image2, classes=[0], conf=self.conf_threshold, verbose=False)

        # Todas las personas detectadas, marcando cuáles están en la ROI
        det1 = self.get_person_detections(res1, self.roi_cam1, w1, h1)
        det2 = self.get_person_detections(res2, roi_cam2, w2, h2)

        # Vídeo anotado para visualización. Solo se dibuja si alguien está suscrito.
        if self.debug_pub1.get_subscription_count() > 0:
            dbg1 = self.draw_debug(cv_image1, det1, self.roi_cam1, 'cam1 (/oak/rgb/image_raw)')
            out1 = self.bridge.cv2_to_imgmsg(dbg1, encoding='bgr8')
            out1.header = msg1.header
            self.debug_pub1.publish(out1)
        if self.debug_pub2.get_subscription_count() > 0:
            dbg2 = self.draw_debug(cv_image2, det2, roi_cam2, 'cam2 (/image_raw)')
            out2 = self.bridge.cv2_to_imgmsg(dbg2, encoding='bgr8')
            out2.header = msg2.header
            self.debug_pub2.publish(out2)

        person_in_cam1 = any(d[5] for d in det1)
        person_in_cam2 = any(d[5] for d in det2)
        current_person_present = person_in_cam1 and person_in_cam2

        # Calcular tiempo transcurrido desde la última vez que se guardó
        current_time = self.get_clock().now()
        time_since_last_save = (current_time - self.last_save_time).nanoseconds / 1e9

        # Lógica de disparo: solo si hay alguien en la ROI de AMBAS cámaras AHORA, NO lo había en el frame anterior,
        # y ha pasado el tiempo de enfriamiento.
        if current_person_present:
            if not self.person_was_present and (time_since_last_save > self.cooldown_seconds):
                # Solo se guarda lo de dentro de la caja: la persona en la ROI de cada cámara
                # (si hay varias en la ROI, la de mayor confianza). Sin cajas dibujadas.
                crop1 = self.crop_detection(cv_image1, self.pick_target(det1), self.crop_margin)
                crop2 = self.crop_detection(cv_image2, self.pick_target(det2), self.crop_margin)

                if crop1 is None or crop2 is None:
                    self.get_logger().warn("Recorte vacío (caja degenerada); no se guarda esta detección.")
                else:
                    self.frame_count += 1
                    n = self.frame_count

                    # Una carpeta por detección: saved_frames/paciente1, paciente2, ...
                    patient_dir = os.path.join(self.save_dir, f"paciente{n}")
                    os.makedirs(patient_dir, exist_ok=True)
                    cv2.imwrite(os.path.join(patient_dir, f"paciente{n}_cam1.jpg"), crop1)
                    cv2.imwrite(os.path.join(patient_dir, f"paciente{n}_cam2.jpg"), crop2)

                    self.get_logger().info(
                        f"¡Nueva persona detectada! Recortes {crop1.shape[1]}x{crop1.shape[0]} (cam1) y "
                        f"{crop2.shape[1]}x{crop2.shape[0]} (cam2) guardados en {patient_dir}")

                    # Actualizar el reloj
                    self.last_save_time = current_time

            # Actualizar el estado indicando que la zona está ocupada
            self.person_was_present = True
        else:
            # Si no hay nadie, reseteamos el estado para que la próxima vez vuelva a disparar
            self.person_was_present = False


def main(args=None):
    rclpy.init(args=args)
    node = DualCameraPersonDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()