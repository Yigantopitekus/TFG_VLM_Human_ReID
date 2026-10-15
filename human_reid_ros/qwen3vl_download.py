"""Instala (descarga UNA vez) los pesos de Qwen3-VL-8B desde Hugging Face.

Es el único punto del proyecto que accede a internet. Después, ni el
servidor ni el nodo ROS descargan nada.

Uso:
    python -m human_reid_ros.qwen3vl_download
    python -m human_reid_ros.qwen3vl_download --local-dir /ruta/modelo
"""
import argparse
import os

from .vlms.common import DEFAULT_MODEL_PATH, REPO_ID


def main(args=None):

    parser = argparse.ArgumentParser(
        description="Descarga Qwen3-VL-8B-Instruct desde Hugging Face (~17 GB)."
    )
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--local-dir", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--revision", default=None,
                        help="Rama, tag o commit (opcional).")
    parser.add_argument("--token", default=None,
                        help="Token de HF (opcional; también vale `hf auth login`).")
    opts = parser.parse_args(args)

    # Aquí sí queremos red, aunque el entorno tenga el modo offline activado.
    os.environ["HF_HUB_OFFLINE"] = "0"

    from huggingface_hub import snapshot_download

    local_dir = os.path.expanduser(opts.local_dir)
    print(f"Descargando {opts.repo_id} en {local_dir} ...")

    snapshot_download(
        repo_id=opts.repo_id,
        local_dir=local_dir,
        revision=opts.revision,
        token=opts.token,
    )

    if not os.path.isfile(os.path.join(local_dir, "config.json")):
        raise SystemExit("La descarga terminó pero falta config.json.")

    print(f"Modelo instalado en {local_dir}")
    print("Ya puedes lanzar:  python -m human_reid_ros.qwen3vl_server")


if __name__ == "__main__":
    main()
