cmake_minimum_required(VERSION 3.8)
project(human_reid_ros)

find_package(ament_cmake REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Python3: es el mismo find_package que usa rosidl_generator_py internamente,
# así la versión de Python de las interfaces y la del módulo coinciden siempre.
# Si tienes conda/venv activo al compilar, desactívalo o fuerza el intérprete:
#   colcon build --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(HUMAN_REID_PYTHON_DIR
  "lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages")

# 1. Interfaz personalizada (srv/VlmInference.srv)
rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/VlmInference.srv"
  DEPENDENCIES sensor_msgs
)

# 2. Módulo Python (image_utils, qwen3vl/, ...) junto a las interfaces generadas.
#    Es recursivo: cada backend vive en su subpaquete human_reid_ros/<modelo>/
install(DIRECTORY human_reid_ros/
  DESTINATION ${HUMAN_REID_PYTHON_DIR}/${PROJECT_NAME}
  FILES_MATCHING PATTERN "*.py"
)

# 3. Nodos ejecutables -> `ros2 run human_reid_ros <nombre>.py`
#    IMPORTANTE: cada script necesita permiso de ejecución (chmod +x) en el repo.
#    Con `colcon build --symlink-install` se instala un enlace al fichero original,
#    y si éste no es ejecutable `ros2 run` no lo encuentra.
#    Todos acaban planos en lib/${PROJECT_NAME}, por eso cada backend nombra sus nodos
#    <modelo>_service_node.py / <modelo>_reid_node.py (reid.launch.py usa esa convención).
install(PROGRAMS
  # --- Backend Qwen3-VL ---
  human_reid_ros/qwen3vl/qwen3vl_service_node.py
  human_reid_ros/qwen3vl/qwen3vl_reid_node.py
  # --- Auxiliares (independientes de cualquier modelo) ---
  human_reid_ros/image_publisher.py
  human_reid_ros/person_detector.py
  DESTINATION lib/${PROJECT_NAME}
)

# 4. Launch files y parámetros por modelo (config/<modelo>.yaml)
install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()