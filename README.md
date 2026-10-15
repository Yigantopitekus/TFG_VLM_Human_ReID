# TFG_VLM_Human_ReID

Este paquete de ROS 2 implementa un sistema de Re-Identificación de Personas (Human ReID) utilizando Modelos de Visión-Lenguaje (VLMs), específicamente **Qwen3-VL**. 

La arquitectura está diseñada íntegramente sobre el ecosistema de ROS 2, eliminando dependencias externas de servidores HTTP y optimizando el uso de VRAM al cargar los pesos del modelo una única vez en un nodo servidor dedicado.

## 🏗️ Arquitectura del Sistema

El paquete se compone de dos nodos principales que se comunican mediante un servicio ROS personalizado:

1. **`qwen3vl_service_node` (Servidor VLM):** 
   - Inicializa y mantiene el motor de Qwen3-VL activo en memoria (VRAM).
   - Levanta el servicio `vlm_inference`.
   - Espera peticiones síncronas de otros nodos para procesar imágenes con texto.

2. **`qwen3vl_reid_node` (Cliente ReID):** 
   - Se suscribe a un topic de imágenes (por defecto `/camera/image_raw`).
   - Adjunta el *prompt* especializado en ReID a la imagen.
   - Actúa como cliente del servicio `vlm_inference`, enviando la petición asíncrona y gestionando la respuesta del modelo.

### Interfaz de Servicio (`VlmInference.srv`)
```text
sensor_msgs/Image image
string prompt
---
string response
