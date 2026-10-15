# TFG_VLM_Human_ReID

Re-identificación de personas con un modelo visión-lenguaje (VLM) a partir de imágenes de referencia.
Paquete ROS 2 (`ament_python`) llamado **`human_reid_ros`**.

El único modelo usado es **Qwen3-VL-8B-Instruct** (`Qwen/Qwen3-VL-8B-Instruct`).

## Arquitectura

```
┌──────────────────────┐   HTTP (prompts + imágenes)   ┌──────────────────────────┐
│  qwen3vl_reid_node   │ ────────────────────────────► │  qwen3vl_server          │
│  (ROS 2, sin torch,  │ ◄──────────────────────────── │  (proceso independiente, │
│   sin pesos)         │        texto generado         │   carga Qwen3-VL-8B)     │
└──────────────────────┘                               └──────────────────────────┘
```

* **Los pesos se descargan una sola vez** con `qwen3vl_download` desde Hugging Face.
* Ni el servidor ni el nodo descargan nada al ejecutarse: el motor carga con
  `local_files_only=True` y modo offline de HF. Si faltan los pesos, falla con un mensaje claro.
* El nodo ROS en modo `server` no importa torch/transformers y no necesita GPU.
* El servidor habla un subconjunto de la API OpenAI (`/v1/chat/completions`), así que el nodo
  también funciona contra otros servidores compatibles (p. ej. `vllm serve Qwen/Qwen3-VL-8B-Instruct`).

## Estructura

```
├── package.xml · setup.py · setup.cfg · resource/
├── requirements.txt           # cliente / nodo ROS (ligero)
├── requirements-server.txt    # servidor del modelo (torch, transformers>=4.57, ...)
├── test/                      # tests (sin ROS ni GPU)
└── human_reid_ros/
    ├── qwen3vl_reid_node.py   # nodo ROS (backend server | local)
    ├── image_publisher.py     # consola interactiva: ref / crowd / exit
    ├── qwen3vl_server.py      # servidor independiente del modelo
    ├── qwen3vl_download.py    # instalación de los pesos desde Hugging Face
    └── vlms/
        ├── common.py          # constantes + codificación de imágenes/mensajes
        ├── prompts.py         # prompts de re-ID
        ├── qwen3vl_reid.py    # lógica re-ID (elige backend)
        ├── qwen3vl_remote.py  # cliente HTTP (stdlib)
        └── qwen3vl_engine.py  # inferencia con transformers (único módulo con torch)
```

## 1. Instalar el modelo (una vez)

Entorno virtual del **servidor** (necesita GPU con ~8 GB+ en 4 bits; ~17 GB de disco para los pesos):

```bash
python3 -m venv ~/qwen3vl_env
source ~/qwen3vl_env/bin/activate
python -m pip install -r requirements-server.txt

# Descarga única desde Hugging Face (~17 GB) a ~/TFG/vlm_human_reid/models/Qwen3-VL-8B-Instruct
python -m human_reid_ros.qwen3vl_download
# Opciones: --local-dir RUTA  --token HF_TOKEN  --revision COMMIT
```

## 2. Lanzar el servidor del modelo (proceso independiente)

```bash
source ~/qwen3vl_env/bin/activate
python -m human_reid_ros.qwen3vl_server --port 8000
# Opciones: --model-path RUTA  --quantization 4bit|8bit|none  --host 0.0.0.0
```

Se ejecuta desde la raíz del repo (o con el paquete instalado tras `colcon build`). No necesita ROS.
`--host 0.0.0.0` permite que el nodo ROS esté en otra máquina. Comprobación: `curl http://127.0.0.1:8000/health`.

## 3. Compilar y ejecutar el nodo ROS

```bash
cd ~/ros2_ws          # el repo va dentro de src/
colcon build --packages-select human_reid_ros
source install/setup.bash

# Terminal 1 (no hace falta el entorno del modelo)
ros2 run human_reid_ros qwen3vl_reid_node --ros-args -p server_url:=http://127.0.0.1:8000

# Terminal 2
ros2 run human_reid_ros image_publisher --ros-args \
  -p reference_dir:=~/mis_imagenes/references \
  -p crowd_image_path:=~/mis_imagenes/crowd/crowd1.jpg
```

### Parámetros de `qwen3vl_reid_node`

| Parámetro         | Por defecto                  | Descripción                                             |
|-------------------|------------------------------|---------------------------------------------------------|
| `backend`         | `server`                     | `server` = modelo en proceso aparte · `local` = en el propio nodo |
| `server_url`      | `http://127.0.0.1:8000`      | URL del servidor (backend `server`)                     |
| `request_timeout` | `300.0`                      | Timeout por petición, en segundos                       |
| `model_path`      | `~/TFG/vlm_human_reid/models/Qwen3-VL-8B-Instruct` | Pesos instalados (backend `local`) |
| `quantization`    | `4bit`                       | `4bit` · `8bit` · `none` (backend `local`)              |

`backend:=local` carga el modelo dentro del nodo (necesita el entorno `~/qwen3vl_env` activado), pero **sigue sin descargar nada**: solo lee los pesos ya instalados.

### Topics

| Topic                   | Tipo                | Sentido                          |
|-------------------------|---------------------|----------------------------------|
| `/human_reid/reference` | `sensor_msgs/Image` | entrada (5 referencias)          |
| `/human_reid/crowd`     | `sensor_msgs/Image` | entrada (imagen query)           |
| `/human_reid/result`    | `std_msgs/String`   | salida (`MATCH` / `NO MATCH`)    |

## Tests

```bash
python3 -m pytest test
```

Incluye una prueba extremo a extremo servidor↔cliente con un motor simulado, y comprueba que el
cliente no importa torch y que el motor nunca descarga pesos.
