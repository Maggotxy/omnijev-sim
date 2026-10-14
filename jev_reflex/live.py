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
from .scenario import FOCUSED, Scenario
from .bank import validate_update

HTML = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>OmniJev 条件反射仿真</title><style>
*{box-sizing:border-box}body{margin:0;background:#0b111b;color:#e9f1fc;font:16px system-ui,sans-serif}main{max-width:1240px;margin:auto;padding:24px}h1{font-size:30px;margin:0 0 8px}p{color:#a9b9d0;line-height:1.7}section{background:#141f30;border:1px solid #25364b;border-radius:16px;padding:18px;margin:16px 0}.controls{display:flex;gap:10px;flex-wrap:wrap}button{border:1px solid #42617c;background:#203449;color:#f1f7fe;border-radius:9px;padding:12px 16px;font:inherit;cursor:pointer}button:hover{background:#2c4963}button:disabled{opacity:.4;cursor:wait}.primary{background:#236445;border-color:#66cfa1}.stop{background:#622b38;border-color:#c37289}img{width:100%;display:block;background:#0b111b;border-radius:12px}#status{font-size:19px;color:#8ce2b4}#msg{min-height:25px;color:#ffd391}.note{font-size:14px}#metrics{display:flex;gap:20px;flex-wrap:wrap;color:#a7c8ed}a{color:#89baf6}
</style><main><h1>大脑与小脑：反应速度仿真</h1><p>选择纯视觉大脑或视觉大脑＋小脑。两组都有刹停、退让、抬起和修正能力；突发变化时观察谁实际下达命令、机械臂何时运动。等待模型期间物理世界继续运行。</p>
<section><div class="controls"><button class="primary" onclick="act('start')">启动仿真</button><button id="reflexOn" onclick="act('enable')">开启小脑反射</button><button id="reflexOff" onclick="act('disable')">关闭小脑反射</button><button class="stop" onclick="act('stop')">停止并保存</button><button onclick="act('vlm')">选择纯视觉大脑</button><button onclick="act('vlm_omni')">选择视觉大脑＋OmniJev</button></div><p id="status">待启动</p><div id="metrics"></div><p id="cognition"></p><p id="judgment"></p><div id="reflexes" class="controls"></div></section>
<img id="camera" src="/frame.jpg" alt="实时物理仿真与模型观察画面">
<section><div class="controls"><button onclick="act('hand_close')">人手靠近夹爪</button><button onclick="act('slider_fast')">物体快速横穿</button><button onclick="act('target_jump')">目标突然移位</button><button onclick="act('occlusion_brief')">短暂遮挡相机</button><button onclick="act('random')">切换随机干扰</button><button onclick="act('reset')">重置场景</button></div><p id="msg"></p>
<p class="note">左侧是观察相机，右侧是模型实际看到的画面。橙圈来自机器人自身位置，蓝圈来自原运动计划。黄色 WATCHDOG HOLD 表示软件过期保护，不能算模型反射。每次环境干扰、模型答复、电机动作和真实接触都会保存。</p></section>
<section><p>写一句反射条件，立即加入运行中的反射银行。</p><textarea id="condition" rows="3" style="width:100%;background:#0b111b;color:#e9f1fc;font:inherit;padding:12px;border:1px solid #42617c" placeholder="例如：有人手接近夹爪时，退回一小段"></textarea><div class="controls"><select id="reflexAction" style="font:inherit;padding:10px"><option value="retreat">退让</option><option value="brake">刹停</option><option value="lift">抬起</option><option value="left">向左修正</option><option value="right">向右修正</option></select><label>触发概率阈值 <input id="threshold" type="number" value="0.75" min="0.5" max="0.98" step="0.05" style="width:80px;padding:10px"></label><button onclick="addReflex()">加入反射</button></div></section>
<p class="note">小脑使用现有云端 OmniJev 0.8B。视觉大脑使用 g.92.run 的 Gemini 3.8 Flash，两组使用同一份视觉大脑指令，并允许它直接选择动作；组合组额外启用独立小脑。两组切换需先停止本轮。每轮最多运行 5 分钟；所有概率、条件更新、经验反馈和电机动作都保存。</p></main><script>
async function act(kind){try{const r=await fetch('/command',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({kind})});const d=await r.json();document.getElementById('msg').textContent=d.message||d.error||kind}catch(e){document.getElementById('msg').textContent='连接中断：'+e.message}}
async function addReflex(){try{const condition=document.getElementById('condition').value;const action=document.getElementById('reflexAction').value;const threshold=Number(document.getElementById('threshold').value);const r=await fetch('/command',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({kind:'reflex',updates:[{id:'custom_'+Date.now(),condition,action,threshold}]})});const d=await r.json();document.getElementById('msg').textContent=d.message||d.error}catch(e){document.getElementById('msg').textContent=e.message}}
function renderJudgment(s){const p=s.condition_probabilities||{};document.getElementById('judgment').textContent='模型候选 '+(s.reflex_candidate||'等待当前观察')+' · 条件门控 '+(s.reflex_gate===true?'达到阈值':s.reflex_gate===false?'未触发':'等待答复')+' · 危险评分 '+(s.danger_score==null?'—':s.danger_score.toFixed(2));const box=document.getElementById('reflexes');box.replaceChildren();for(const r of (s.bank?.reflexes||[])){const card=document.createElement('div');card.style.cssText='flex:1 1 240px;padding:10px;background:#0b111b;border-radius:8px';const label=document.createElement('div');label.textContent=r.action+' · '+(p[r.id]==null?'—':p[r.id].toFixed(2))+' / 阈值 '+r.threshold.toFixed(2);const meter=document.createElement('meter');meter.min=0;meter.max=1;meter.value=p[r.id]||0;meter.style.width='100%';const condition=document.createElement('div');condition.textContent=r.condition;condition.style.cssText='font-size:12px;color:#a9b9d0;margin-top:5px';card.append(label,meter,condition);box.append(card)}}
async function refresh(){try{const r=await fetch('/state');const s=await r.json();document.getElementById('status').textContent=(s.status==='running'?'正在运行':s.status==='loading'?'正在建立仿真':s.status==='finished'?'本轮已保存':'待启动')+' · '+(s.model_enabled?'小脑反射开启':'小脑反射关闭')+' · 动作 '+(s.action||'—')+' · 命令来源 '+(s.motor_source||'—');document.getElementById('reflexOn').disabled=s.selected_group==='vlm';document.getElementById('reflexOff').disabled=s.selected_group==='vlm';document.getElementById('metrics').textContent='世界 '+(s.world_s||0).toFixed(2)+' 秒　实际 '+(s.wall_s||0).toFixed(2)+' 秒　人手接触 '+(s.human_contact_N||0).toFixed(1)+' N　目标误差 '+((s.goal_distance_m||0)*100).toFixed(1)+' cm　实测速度 '+(s.measured_eef_speed_m_s||0).toFixed(3)+' m/s　'+(s.auto?'随机干扰开启':'');document.getElementById('cognition').textContent='组别 '+(s.selected_group||s.group||'—')+'　大脑 '+(s.brain_status||'共享初始意图')+'　反射银行版本 '+(s.bank_version||0)+'　经验 '+(s.memory_entries||0)+' 条'+(s.task_completion?'　结果 '+(s.task_completion.success?'稳定完成':'未满足连续保持'):'');renderJudgment(s);document.getElementById('camera').src='/frame.jpg?t='+Date.now();if(s.error)document.getElementById('msg').textContent=s.error}catch(e){document.getElementById('status').textContent='仿真连接暂不可用'}setTimeout(refresh,200)}refresh();
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

    def poll(self, world, bank):
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
            if kind in ("hand", "slider", "target", "occlusion", *FOCUSED):
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
            elif kind == "reflex":
                try:
                    invalidate |= bank.update(command["updates"], "operator", sim_s, allow_new=True)
                    event = {"kind": kind, "sim_s": sim_s, "updates": command["updates"], "bank_version": bank.version}
                except ValueError as exc:
                    event = {"kind": kind, "sim_s": sim_s, "status": "rejected", "reason": str(exc)}
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

    def command(self, kind, updates=None):
        if kind in ("omni", "vlm", "vlm_omni"):
            with self.lock:
                if self.running:
                    return {"message": "先停止本轮，再切换实验组"}
                self.args.mode = kind
            return {"message": "下轮使用："+("纯视觉大脑" if kind == "vlm" else "视觉大脑＋OmniJev" if kind == "vlm_omni" else "OmniJev")}
        if kind == "start":
            with self.lock:
                if self.running:
                    return {"message": "仿真已经在运行"}
                self.counter += 1
                output = self.args.out/f"session_{time.time_ns()}"
                seed = self.args.seed
                self.session = LiveHooks(output, seed)
                self.running = True
                thread = threading.Thread(target=self._run, args=(self.session, output, seed), daemon=True)
                thread.start()
            return {"message": "正在启动连续仿真"}
        if not self.running or self.session is None:
            return {"message": "先启动仿真"}
        self.session.commands.put({"kind": kind, "source": "operator", "updates": updates})
        return {"message": "操作已进入物理循环："+kind}

    def _run(self, session, output, seed):
        scenario = replace(Scenario.generate("quiet", seed), kind="combined", timings=())
        try:
            result = episode(self.args, scenario, self.args.mode, output, live=session)
            with session.lock:
                session.state.update(status="finished", saved_path=str(output), final_goal_reached=result["final_goal_reached"], task_completion=result["task_completion"])
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
                state["selected_group"] = demo.args.mode
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
                if not 0 < length <= 4096:
                    raise ValueError("Invalid request size")
                body = json.loads(self.rfile.read(length))
                kind = body.get("kind")
                if kind not in ("start", "stop", "enable", "disable", "hand", "slider", "target", "occlusion", "random", "reset", "reflex", "omni", "vlm", "vlm_omni", *FOCUSED):
                    raise ValueError("Unknown operation")
                updates = body.get("updates")
                if kind == "reflex":
                    if not isinstance(updates, list) or not 1 <= len(updates) <= 3:
                        raise ValueError("Invalid reflex transaction")
                    updates = [validate_update(update, allow_new=True) for update in updates]
                response = demo.command(kind, updates)
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
    parser.add_argument("--mode", choices=["vlm", "vlm_omni"], default="vlm_omni")
    bundled = Path(__file__).resolve().parents[1]/"resources"/"panda"
    parser.add_argument("--asset-root", type=Path, default=bundled)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.image_size, args.api_timeout = 320, 2
    args.max_observation_age, args.command_ttl, args.render_hz = 1.2, 1.4, 8
    args.brain_timeout, args.brain_interval, args.max_brain_age, args.brain_command_ttl = 15, 0, 8, 10
    args.freeze_bank = False
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler(Demo(args)))
    print(f"Interactive simulation ready: http://127.0.0.1:{args.port} (no model calls until Start)", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
