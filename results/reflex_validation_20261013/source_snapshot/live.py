"""Loopback-only interactive simulator. Operator actions never enter model prompts."""
import argparse
from dataclasses import replace
import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import queue
import threading
import time

import numpy as np
from PIL import Image, ImageDraw

from .run import episode, font
from .scenario import Scenario

HTML = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>OmniJev 小脑反射实验</title><style>
*{box-sizing:border-box}body{margin:0;background:#0b111b;color:#e9f1fc;font:16px system-ui,sans-serif}main{max-width:1240px;margin:auto;padding:24px}h1{font-size:30px;margin:0 0 8px}p{color:#a9b9d0;line-height:1.7}section{background:#141f30;border:1px solid #25364b;border-radius:16px;padding:18px;margin:16px 0}.controls{display:flex;gap:10px;flex-wrap:wrap}button{border:1px solid #42617c;background:#203449;color:#f1f7fe;border-radius:9px;padding:12px 16px;font:inherit;cursor:pointer}button:hover{background:#2c4963}button:disabled{opacity:.4;cursor:wait}.primary{background:#236445;border-color:#66cfa1}.stop{background:#622b38;border-color:#c37289}img{width:100%;display:block;background:#0b111b;border-radius:12px}#status{font-size:19px;color:#8ce2b4}#msg{min-height:25px;color:#ffd391}.note{font-size:14px}#metrics{display:flex;gap:20px;flex-wrap:wrap;color:#a7c8ed}a{color:#89baf6}
</style><main><h1>让小脑反射看得见</h1><p>启动后，机械臂按原计划伸向绿色目标。你可以随时改变环境，观察模型是否刹停、退让或修正；等待模型期间物理世界继续运行。</p>
<section><div class="controls"><button class="primary" onclick="act('start')">启动仿真</button><button onclick="act('enable')">开启模型反射</button><button onclick="act('disable')">关闭模型反射</button><button class="stop" onclick="act('stop')">停止并保存</button></div><p id="status">待启动</p><div id="metrics"></div></section>
<img id="camera" src="/frame.jpg" alt="实时物理仿真与模型观察画面">
<section><div class="controls"><button onclick="act('hand')">突然伸入人手</button><button onclick="act('slider')">推一下黄色物体</button><button onclick="act('target')">挪动绿色目标</button><button onclick="act('occlusion')">遮挡相机</button><button onclick="act('random')">切换随机干扰</button><button onclick="act('reset')">重置场景</button></div><p id="msg"></p>
<p class="note">左侧是观察相机，右侧是模型实际看到的画面。橙圈来自机器人自身位置，蓝圈来自原运动计划。黄色 WATCHDOG HOLD 表示软件过期保护，不能算模型反射。每次环境干扰、模型答复、电机动作和真实接触都会保存。</p></section>
<p class="note">当前连接云端 OmniJev 0.8B。这是持续反射研究原型，会显示误判、延迟与失效；随机干扰和按钮操作不会提前告诉模型。每轮最多运行 5 分钟，停止后可重新启动。</p></main><script>
async function act(kind){try{const r=await fetch('/command',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({kind})});const d=await r.json();document.getElementById('msg').textContent=d.message||d.error||kind}catch(e){document.getElementById('msg').textContent='连接中断：'+e.message}}
async function refresh(){try{const r=await fetch('/state');const s=await r.json();document.getElementById('status').textContent=(s.status==='running'?'正在运行':s.status==='loading'?'正在建立仿真':s.status==='finished'?'本轮已保存':'待启动')+' · '+(s.model_enabled?'模型反射开启':'模型反射关闭')+' · 动作 '+(s.action||'—');document.getElementById('metrics').textContent='世界 '+(s.world_s||0).toFixed(2)+' 秒　实际 '+(s.wall_s||0).toFixed(2)+' 秒　人手接触 '+(s.human_contact_N||0).toFixed(1)+' N　目标误差 '+((s.goal_distance_m||0)*100).toFixed(1)+' cm　'+(s.auto?'随机干扰开启':'');document.getElementById('camera').src='/frame.jpg?t='+Date.now();if(s.error)document.getElementById('msg').textContent=s.error}catch(e){document.getElementById('status').textContent='仿真连接暂不可用'}setTimeout(refresh,200)}refresh();
</script></html>"""


class LiveHooks:
    def __init__(self, output, seed):
        self.output, self.rng = output, np.random.default_rng(seed+9011)
        self.commands = queue.Queue()
        self.lock = threading.Lock()
        self.enabled, self.stopping, self.auto = True, False, False
        self.next_random = 0
        self.state = {"status": "loading", "model_enabled": True}
        image = Image.new("RGB", (960, 720), (11, 17, 27))
        ImageDraw.Draw(image).text((50, 300), "Starting physical simulator...", fill=(166, 228, 194), font=font(30))
        self.frame = np.asarray(image)

    def update(self, frame, state):
        with self.lock:
            self.frame = frame
            self.state = {**state, "auto": self.auto}

    def poll(self, world):
        invalidate = False
        todo = []
        while not self.commands.empty():
            todo.append(self.commands.get_nowait())
        sim_s = float(world.data.time-world.zero)
        if self.auto and sim_s >= self.next_random:
            todo.append({"kind": str(self.rng.choice(["hand", "slider", "target", "occlusion"])), "source": "random_environment"})
            self.next_random = sim_s+float(self.rng.exponential(4)+1.5)
        for command in todo:
            kind = command["kind"]
            if kind in ("hand", "slider", "target", "occlusion"):
                event = world.inject(kind, self.rng)
            elif kind == "enable":
                self.enabled, invalidate = True, True
                event = {"kind": kind, "sim_s": sim_s}
            elif kind == "disable":
                self.enabled, invalidate = False, True
                event = {"kind": kind, "sim_s": sim_s}
            elif kind == "random":
                self.auto = not self.auto
                self.next_random = sim_s+.5
                event = {"kind": kind, "sim_s": sim_s, "auto": self.auto}
            elif kind == "reset":
                world.reset_live()
                invalidate = True
                event = {"kind": kind, "sim_s": sim_s}
            elif kind == "stop":
                self.stopping = True
                event = {"kind": kind, "sim_s": sim_s}
            else:
                continue
            with open(self.output/"interactive_events.jsonl", "a") as stream:
                stream.write(json.dumps({**event, "source": command.get("source", "operator")})+"\n")
        return invalidate


class Demo:
    def __init__(self, args):
        self.args, self.session = args, None
        self.lock = threading.Lock()
        self.running = False
        self.counter = 0

    def command(self, kind):
        if kind == "start":
            with self.lock:
                if self.running:
                    return {"message": "仿真已经在运行"}
                self.counter += 1
                output = self.args.out/f"session_{time.time_ns()}"
                seed = self.args.seed+self.counter-1
                self.session = LiveHooks(output, seed)
                self.running = True
                thread = threading.Thread(target=self._run, args=(self.session, output, seed), daemon=True)
                thread.start()
            return {"message": "正在启动连续仿真"}
        if not self.running or self.session is None:
            return {"message": "先启动仿真"}
        self.session.commands.put({"kind": kind, "source": "operator"})
        return {"message": "操作已进入物理循环："+kind}

    def _run(self, session, output, seed):
        scenario = replace(Scenario.generate("quiet", seed), kind="combined", timings=())
        try:
            result = episode(self.args, scenario, "reflex", output, live=session)
            with session.lock:
                session.state.update(status="finished", saved_path=str(output), final_goal_reached=result["final_goal_reached"])
        except Exception as exc:
            with session.lock:
                session.state.update(status="finished", error=f"{type(exc).__name__}: {exc}")
            print(f"Live session failed: {type(exc).__name__}: {exc}", flush=True)
        finally:
            with self.lock:
                self.running = False


def handler(demo):
    class Handler(BaseHTTPRequestHandler):
        def send(self, content, mime, code=200):
            self.send_response(code)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/":
                self.send(HTML.encode(), "text/html; charset=utf-8")
            elif path == "/state":
                session = demo.session
                if session:
                    with session.lock:
                        state = dict(session.state)
                else:
                    state = {"status": "ready", "model_enabled": False}
                self.send(json.dumps(state).encode(), "application/json")
            elif path == "/frame.jpg":
                session = demo.session
                if session:
                    with session.lock:
                        frame = session.frame
                else:
                    frame = np.zeros((720, 960, 3), dtype=np.uint8)
                buffer = io.BytesIO()
                Image.fromarray(frame).save(buffer, "JPEG", quality=80)
                self.send(buffer.getvalue(), "image/jpeg")
            else:
                self.send(b"Not found", "text/plain", 404)

        def do_POST(self):
            if self.path != "/command":
                self.send(b"Not found", "text/plain", 404)
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= 1024:
                    raise ValueError("Invalid request size")
                kind = json.loads(self.rfile.read(length)).get("kind")
                if kind not in ("start", "stop", "enable", "disable", "hand", "slider", "target", "occlusion", "random", "reset"):
                    raise ValueError("Unknown operation")
                response = demo.command(kind)
                self.send(json.dumps(response).encode(), "application/json")
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                self.send(json.dumps({"error": str(exc)}).encode(), "application/json", 400)

        def log_message(self, *args):
            pass
    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8790)
    parser.add_argument("--seed", type=int, default=173)
    parser.add_argument("--seconds", type=float, default=300)
    bundled = Path(__file__).resolve().parents[1]/"resources"/"panda"
    parser.add_argument("--asset-root", type=Path, default=bundled)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.image_size, args.api_timeout = 320, 2
    args.max_observation_age, args.command_ttl, args.render_hz = 1.2, 1.4, 8
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler(Demo(args)))
    print(f"Interactive simulation ready: http://127.0.0.1:{args.port} (no model calls until Start)", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
