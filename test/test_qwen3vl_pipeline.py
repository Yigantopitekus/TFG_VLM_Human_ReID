"""Prueba extremo a extremo servidor <-> cliente con un motor simulado.

No necesita ROS, GPU, torch ni pesos: el motor real se sustituye por un
FakeEngine, pero el servidor HTTP, el protocolo y el cliente son los reales.
"""
import ast
import pathlib
import subprocess
import sys
import threading
import urllib.request

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from human_reid_ros.qwen3vl_server import make_server  # noqa: E402
from human_reid_ros.vlms.qwen3vl_reid import Qwen3VLReID  # noqa: E402


class FakeEngine:
    """Registra lo que recibe y responde de forma determinista."""

    def __init__(self):
        self.calls = []

    def chat(self, messages, max_new_tokens=512):
        self.calls.append((messages, max_new_tokens))
        n_images = sum(
            p["type"] == "image" for m in messages for p in m["content"]
        )
        last_user = messages[-1]["content"][-1]["text"]
        if last_user.startswith("Analyze"):
            return f"MATCH (vi {n_images} imágenes en total)"
        return "Blue jacket, black jeans, white sneakers."


@pytest.fixture()
def served():
    engine = FakeEngine()
    server = make_server(engine, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    yield engine, url
    server.shutdown()
    server.server_close()


def _bgr_image(color, size=(120, 60)):
    img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    img[:, :] = color
    return img


def test_health(served):
    _, url = served
    with urllib.request.urlopen(url + "/health") as r:
        assert r.status == 200


def test_full_reid_cycle_over_http(served):
    engine, url = served
    qwen = Qwen3VLReID(backend="server", server_url=url)
    assert qwen.remote.is_ready()

    refs = [_bgr_image((255, 0, 0)) for _ in range(5)]  # BGR azul puro
    profile = qwen.describe_references(refs)
    assert profile == "Blue jacket, black jeans, white sneakers."

    messages, max_tokens = engine.calls[0]
    assert max_tokens == 512
    assert len(messages) == 1
    assert sum(p["type"] == "image" for p in messages[0]["content"]) == 5
    # BGR -> RGB: (255,0,0) en BGR debe llegar como azul en RGB (0,0,255).
    first = messages[0]["content"][0]["image"]
    r, g, b = first.getpixel((5, 5))
    assert b > 200 and r < 60 and g < 60

    result = qwen.describe_query(_bgr_image((0, 255, 0)))
    assert result.startswith("MATCH")

    # La segunda petición lleva el historial: 5 refs + perfil + query.
    messages, _ = engine.calls[1]
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert messages[1]["content"][0]["text"] == profile
    assert "6 imágenes" in result


def test_query_before_references_fails(served):
    _, url = served
    qwen = Qwen3VLReID(backend="server", server_url=url)
    with pytest.raises(RuntimeError):
        qwen.describe_query(_bgr_image((0, 0, 0)))


def test_server_down_gives_clear_error():
    qwen = Qwen3VLReID(backend="server", server_url="http://127.0.0.1:9",
                       timeout=2)
    assert not qwen.remote.is_ready(timeout=0.5)
    with pytest.raises(RuntimeError, match="qwen3vl_server"):
        qwen.describe_references([_bgr_image((1, 2, 3))])


def test_remote_url_images_are_rejected(served):
    _, url = served
    import json
    body = json.dumps({"messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "http://example.com/a.jpg"}}
    ]}]}).encode()
    req = urllib.request.Request(url + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(req)
    assert exc.value.code == 400


def test_client_does_not_import_torch():
    code = (
        "import sys; "
        "import human_reid_ros.vlms.qwen3vl_reid; "
        "import human_reid_ros.vlms.qwen3vl_remote; "
        "import human_reid_ros.qwen3vl_server; "
        "import human_reid_ros.qwen3vl_download; "
        "sys.exit(1 if 'torch' in sys.modules or 'transformers' in sys.modules else 0)"
    )
    proc = subprocess.run([sys.executable, "-c", code], cwd=ROOT)
    assert proc.returncode == 0, "el cliente no debe importar torch/transformers"


def test_engine_never_downloads_weights():
    """Toda llamada a from_pretrained del motor debe ser local_files_only."""
    src = (ROOT / "human_reid_ros/vlms/qwen3vl_engine.py").read_text()
    assert 'os.environ.setdefault("HF_HUB_OFFLINE", "1")' in src
    tree = ast.parse(src)
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call)
             and getattr(n.func, "attr", "") == "from_pretrained"]
    assert calls
    for call in calls:
        kw = {k.arg for k in call.keywords}
        # kwargs puede venir de un dict que ya contiene local_files_only.
        has_flag = "local_files_only" in kw or any(
            k.arg is None for k in call.keywords)
        assert has_flag
    assert "local_files_only=True" in src


def test_only_qwen3vl_remains():
    names = {p.name for p in (ROOT / "human_reid_ros").rglob("*.py")}
    for old in ("qwen25_model.py", "qwen35_model.py", "moondream_model.py",
                "florence_model.py", "qwen25_reid_node.py",
                "qwen35_reid_node.py", "moondream_reid_node.py"):
        assert old not in names
