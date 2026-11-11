"""Lanza el servicio VLM y el nodo ReID de un modelo.

    ros2 launch human_reid_ros reid.launch.py                      # qwen3vl (por defecto)
    ros2 launch human_reid_ros reid.launch.py model:=qwen3vl model_id:=Qwen/Qwen3-VL-4B-Instruct
    ros2 launch human_reid_ros reid.launch.py params_file:=/ruta/mis_params.yaml

NO lanza image_publisher: es una consola interactiva (lee 'ref'/'crowd' por teclado) y
`ros2 launch` no reenvía stdin. Se lanza aparte, en otra terminal:

    ros2 run human_reid_ros image_publisher.py

Convención para añadir un modelo <m> (el launcher lo descubre solo):
    config/<m>.yaml                      parámetros, con claves <m>_service_node y <m>_reid_node
    human_reid_ros/<m>/<m>_service_node.py   ofrece el servicio /vlm_inference
    human_reid_ros/<m>/<m>_reid_node.py      cliente ReID
y registrar los dos .py en install(PROGRAMS ...) del CMakeLists.txt.
Los dos modelos comparten el nombre de servicio '/vlm_inference': se lanza uno cada vez.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

PACKAGE = "human_reid_ros"
DEFAULT_MODEL = "qwen3vl"


def _config_dir() -> str:
    return os.path.join(get_package_share_directory(PACKAGE), "config")


def _available_models() -> list:
    """Un modelo está disponible si existe config/<modelo>.yaml."""
    return sorted(f[: -len(".yaml")] for f in os.listdir(_config_dir()) if f.endswith(".yaml"))


def _launch_setup(context, *args, **kwargs):
    model = LaunchConfiguration("model").perform(context)
    models = _available_models()
    if model not in models:
        raise RuntimeError(
            f"Modelo '{model}' desconocido. Disponibles: {', '.join(models) or '(ninguno)'}."
        )

    params_file = LaunchConfiguration("params_file").perform(context) or os.path.join(
        _config_dir(), f"{model}.yaml"
    )
    if not os.path.isfile(params_file):
        raise RuntimeError(f"No existe el fichero de parámetros: {params_file}")

    service_params = [params_file]
    model_id = LaunchConfiguration("model_id").perform(context)
    if model_id:  # el override suelto va después del YAML, así que tiene prioridad
        service_params.append({"model_id": model_id})

    # Si cualquiera de los dos muere (p. ej. CUDA out of memory al cargar el modelo),
    # se apaga todo: el cliente se quedaría esperando un servicio que ya no existe.
    stop_all = Shutdown(reason=f"Un nodo de '{model}' ha terminado")

    service = Node(
        package=PACKAGE,
        executable=f"{model}_service_node.py",
        name=f"{model}_service_node",
        output="screen",
        emulate_tty=True,
        parameters=service_params,
        on_exit=stop_all,
    )
    reid = Node(
        package=PACKAGE,
        executable=f"{model}_reid_node.py",
        name=f"{model}_reid_node",
        output="screen",
        emulate_tty=True,
        parameters=[params_file],
        on_exit=stop_all,
    )
    return [service, reid]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "model",
                default_value=DEFAULT_MODEL,
                description="Backend VLM a lanzar (uno por cada config/<modelo>.yaml).",
            ),
            DeclareLaunchArgument(
                "params_file",
                default_value="",
                description="YAML de parámetros alternativo (por defecto config/<model>.yaml).",
            ),
            DeclareLaunchArgument(
                "model_id",
                default_value="",
                description="Override opcional del model_id de Hugging Face del servicio.",
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
