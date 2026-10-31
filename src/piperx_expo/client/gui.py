"""本机浏览器客户端：默认模拟，真实控制另需--allow-motion和现场确认。"""

import argparse
import io
import json
import mimetypes
import re
import secrets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import h5py
from PIL import Image

from piperx_expo.client.gui_control import GUIController
from piperx_expo.config import load_config
from piperx_expo.observation import IMAGE_KEYS

WEB = Path(__file__).with_name("web")


def create_server(controller, host="127.0.0.1", port=8767):
    if host not in {"127.0.0.1", "localhost"}:
        raise ValueError("控制界面仅监听本机；远程访问请使用SSH端口转发")
    token = secrets.token_urlsafe(32)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send_bytes(self, data, content_type="application/json", code=200, download=False):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; frame-ancestors 'none'",
            )
            if download:
                self.send_header("Content-Disposition", 'attachment; filename="episode.hdf5"')
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def send_archive(self, path):
            # 原始双相机档案可能很大，逐块发送避免下载时再占用整条轨迹的内存。
            self.send_response(200)
            self.send_header("Content-Type", "application/x-hdf5")
            self.send_header("Content-Length", str(path.stat().st_size))
            self.send_header("Content-Disposition", 'attachment; filename="episode.hdf5"')
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                with path.open("rb") as file:
                    while chunk := file.read(1024 * 1024):
                        self.wfile.write(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def send_json(self, value, code=200):
            self.send_bytes(json.dumps(value, ensure_ascii=False, allow_nan=False).encode(), code=code)

        def valid_host(self):
            return self.headers.get("Host") in {
                f"127.0.0.1:{self.server.server_port}",
                f"localhost:{self.server.server_port}",
            }

        def do_GET(self):
            if not self.valid_host():
                self.send_json({"error": "Host无效"}, 403)
                return
            try:
                parsed = urlparse(self.path)
                path = parsed.path
                if path == "/api/state":
                    # 心跳必须来自已打开的控制页，跨站图片请求不能延长运动许可。
                    if self.headers.get("X-Expo-Token") != token:
                        self.send_json({"error": "会话令牌无效"}, 403)
                        return
                    self.send_json(controller.state())
                elif path == "/api/history":
                    self.send_json(controller.history())
                elif path == "/api/frame":
                    camera = int(parse_qs(parsed.query).get("camera", ["0"])[0])
                    if camera not in (0, 1) or controller.latest_obs is None:
                        self.send_json({"error": "暂无画面"}, 404)
                        return
                    output = io.BytesIO()
                    Image.fromarray(controller.latest_obs[IMAGE_KEYS[camera]]).save(
                        output, "JPEG", quality=85
                    )
                    self.send_bytes(output.getvalue(), "image/jpeg")
                elif path.startswith("/api/episode/"):
                    parts = path.split("/")
                    if len(parts) != 5 or not re.fullmatch(r"[0-9a-f]{32}", parts[3]):
                        raise ValueError("轨迹ID无效")
                    folder = controller.root / parts[3]
                    if parts[4] == "metadata":
                        self.send_bytes((folder / "metadata.json").read_bytes())
                    elif parts[4] == "download":
                        self.send_archive(folder / "episode.hdf5")
                    elif parts[4] == "frame":
                        query = parse_qs(parsed.query)
                        index, camera = int(query.get("index", ["0"])[0]), int(query.get("camera", ["0"])[0])
                        if camera not in (0, 1) or index < 0:
                            raise ValueError("帧位置无效")
                        with h5py.File(folder / "episode.hdf5", "r") as file:
                            rgb = file[f"observations/{IMAGE_KEYS[camera]}"][index]
                        output = io.BytesIO()
                        Image.fromarray(rgb).save(output, "JPEG", quality=85)
                        self.send_bytes(output.getvalue(), "image/jpeg")
                    else:
                        raise ValueError("未知资源")
                elif path in {"/", "/index.html", "/style.css", "/app.js"}:
                    file = WEB / ("index.html" if path == "/" else path[1:])
                    data = file.read_bytes().replace(b"__EXPO_TOKEN__", token.encode())
                    self.send_bytes(data, mimetypes.guess_type(file.name)[0] or "text/plain")
                else:
                    self.send_json({"error": "页面不存在"}, 404)
            except (ValueError, KeyError, IndexError, OSError) as error:
                self.send_json({"error": str(error)}, 400)

        def do_POST(self):
            expected = {
                f"http://127.0.0.1:{self.server.server_port}",
                f"http://localhost:{self.server.server_port}",
            }
            if (
                not self.valid_host()
                or self.headers.get("X-Expo-Token") != token
                or self.headers.get("Origin") not in expected
            ):
                self.send_json({"error": "操作来源校验失败"}, 403)
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 8192 or not self.path.startswith("/api/command/"):
                    raise ValueError("请求格式或长度无效")
                values = json.loads(self.rfile.read(size))
                if not isinstance(values, dict):
                    raise ValueError("请求必须为对象")
                controller.command(self.path.rsplit("/", 1)[-1], values)
                self.send_json({"ok": True})
            except (ValueError, TypeError, RuntimeError, KeyError) as error:
                self.send_json({"error": str(error)}, 400)

    return ThreadingHTTPServer((host, port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/client/gui.mock.yaml")
    parser.add_argument("--allow-motion", action="store_true")
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--record-dir", default="runtime/gui_episodes")
    args = parser.parse_args()
    controller = GUIController(load_config(args.config), args.record_dir, args.allow_motion)
    server = create_server(controller, port=args.port)
    print(
        f"浏览器地址：http://127.0.0.1:{server.server_port}（backend={controller.config['backend']}）",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        controller.stop()
        # shutdown不能从serve_forever线程调用；此处已经退出服务循环。
        server.server_close()
        if controller.worker:
            controller.worker.join(timeout=5)


if __name__ == "__main__":
    main()
