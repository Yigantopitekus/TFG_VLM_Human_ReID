"""Conversión sensor_msgs/Image <-> numpy sin depender de cv_bridge/OpenCV."""
import numpy as np

# encoding -> (canales, orden de canales hacia RGB)
_ENCODINGS = {
    "rgb8": (3, None),
    "bgr8": (3, [2, 1, 0]),
    "rgba8": (4, [0, 1, 2]),
    "bgra8": (4, [2, 1, 0]),
    "mono8": (1, None),
}


def imgmsg_to_rgb(msg) -> np.ndarray:
    """Devuelve un array uint8 HxWx3 en RGB a partir de un sensor_msgs/Image."""
    if msg.encoding not in _ENCODINGS:
        raise ValueError(
            f"Encoding no soportado: '{msg.encoding}' "
            f"(soportados: {', '.join(_ENCODINGS)})"
        )
    channels, order = _ENCODINGS[msg.encoding]
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    # 'step' puede incluir padding al final de cada fila
    rows = buf[: msg.height * msg.step].reshape(msg.height, msg.step)
    img = rows[:, : msg.width * channels].reshape(msg.height, msg.width, channels)
    if channels == 1:
        return np.repeat(img, 3, axis=2)
    if order is not None:
        img = img[:, :, order]
    return np.ascontiguousarray(img)


def rgb_to_imgmsg(rgb: np.ndarray, stamp=None, frame_id: str = ""):
    """Crea un sensor_msgs/Image (rgb8) a partir de un array uint8 HxWx3."""
    from sensor_msgs.msg import Image

    msg = Image()
    if stamp is not None:
        msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height, msg.width = int(rgb.shape[0]), int(rgb.shape[1])
    msg.encoding = "rgb8"
    msg.is_bigendian = 0
    msg.step = msg.width * 3
    msg.data = np.ascontiguousarray(rgb, dtype=np.uint8).tobytes()
    return msg
