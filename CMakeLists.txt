cmake_minimum_required(VERSION 3.8)
project(human_reid_ros)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# 1. Compilar el servicio personalizado
rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/VlmInference.srv"
  DEPENDENCIES sensor_msgs
)

# 2. Descubrir la ruta de instalación de Python en el sistema actual
execute_process(
  COMMAND python3 -c "import sys; print(f'lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages')"
  OUTPUT_VARIABLE PYTHON_INSTALL_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

# 3. Instalar la librería de Python manualmente (Evita el conflicto de ament_python)
install(DIRECTORY human_reid_ros/
  DESTINATION ${PYTHON_INSTALL_DIR}/human_reid_ros
  FILES_MATCHING PATTERN "*.py"
)

# 4. Instalar los nodos ejecutables
install(PROGRAMS
  human_reid_ros/qwen3vl_service_node.py
  human_reid_ros/qwen3vl_reid_node.py
  human_reid_ros/image_publisher.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()