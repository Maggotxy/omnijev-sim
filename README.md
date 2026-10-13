# OmniJev Closed-Loop Robot Simulation

现有 OmniJev 充当局部决策小脑，g.92.run 的 Gemini 3.8 Flash 在后台充当视觉任务大脑。持续 MuJoCo 物理中可以伸入人手、引入自由物体、移动目标或遮挡相机，观察条件概率、动作仲裁和下一次电机控制。模型权重固定，自然语言条件、经验检索与阈值在运行时管理。

## 连续反射仿真

物理以 500 Hz 运行，电机控制为 20 Hz，模型和渲染异步处理最新观察。推理期间世界继续运动；过期软件保护与模型动作分别显示和统计。Panda 资源已随仓库提供，这个模块不依赖 LIBERO 或本地模型权重。

- [启动方法与实时交互说明](docs/reflex.md)
- [最新两组配对结果与全部失败](docs/reflex-runtime-results-20261013.md)
- [实验设计与感知边界](docs/reflex-design.md)
- [Gemini 3.8 Flash 接口实测](docs/model-selection.md)
- [最终交互操作验收证据](results/reflex_runtime_ui_release_20261013/qa_receipt.json)

```bash
pip install -r requirements-reflex.txt
export MUJOCO_GL=osmesa
export OMNIJEV_KEY=your_authorized_key
export GEMINI_API_KEY=your_authorized_relay_key
python -m jev_reflex.live --out out/reflex_interactive --port 8790
```

Linux 下启动后访问 <http://127.0.0.1:8790>。Windows 连接当前 bxj 部署可运行 `scripts/open-reflex.ps1`。最新六组配对中，视觉大脑组到目标 4/6、纯 OmniJev 2/6，采集到命令中位约 0.89 秒；两组各记录到一次事件后匹配的新命令。模型仍会误判、超时和漏掉干扰，当前数据未建立可靠的通用反射能力。

> **Unofficial research demo.** 模型读取当前仿真画面，输出选择或概率；运动由经典控制器执行。历史抓放实验使用仿真位姿辅助，并按仿真时间回放。新连续反射实验的模型输入仅含画面、触觉与当前动作，录像按真实墙钟回放；两套结果各自记录限制。

## 历史抓放三组实验

| 组别 | 规划 | 动作阶段放行 |
|---|---|---|
| Gemini 2.5 Flash | Gemini 2.5 Flash | Gemini 2.5 Flash |
| Gemini + OmniJev | Gemini 2.5 Flash | OmniJev |
| OmniJev | OmniJev | OmniJev |

三组共享官方初始状态、仿真感知辅助和 20 Hz OSC 技能。模型返回 wait 时保持姿态；初始状态指纹、模型响应、逐步动作和官方成功判定均保存。运行方式、接口状态和实验限制见 [comparison.md](docs/comparison.md)。

模型由环境变量明确指定，并记录实际服务端模型名。这批历史抓放实验使用 `gemini-2.5-flash`。

2026-10-13 实测：同一布局下两个 seed，Gemini + OmniJev 完成 **2/2** 个 episode，Gemini 单独组与 OmniJev 单独组均为 **0/2**。这是使用仿真位姿辅助的小样本结果。查看 [完整结果报告](docs/results-20261013.md)、[seed 1 三栏视频](results/comparison_gemini25_20261013/comparison_seed1.mp4) 和 [seed 7 三栏视频](results/comparison_gemini25_20261013/comparison_seed7.mp4)。

## 亮点 | Highlights

- 🎯 **LIBERO-10 官方任务真闭环成功**："put both the alphabet soup and the tomato sauce in the basket" — 98 次决策、39.6 秒仿真时间、环境官方 `success=True` 判定
- 🧠 **历史 LIBERO 演示**：使用 region-box 指向格式和 LIBERO 风格的问题模板
- 👁️ **语义/几何分层架构**：模型输出物体与区域选择，仿真位姿提供精修，经典控制器执行动作
- 📊 **历史 OmniJev / StepFun 小样本**：格子定位 11/16 vs 13/16、决策 3/7 vs 4/7、延迟 0.53s vs 0.90s；一次闭环中的两个子任务完成数为 2/2 vs 0/2，不能当作独立 episode 成功率
- 🎬 全程 HUD 可视化：每次决策的区域指向、概率条、相位状态机实时可见

## 视频 | Videos

| 视频 | 内容 |
|---|---|
| [videos/libero_omnijev_success.mp4](videos/libero_omnijev_success.mp4) | LIBERO-10 双物体入篮，真闭环成功（实时速度回放） |
| [videos/demo_pretty_omnijev.mp4](videos/demo_pretty_omnijev.mp4) | 自建高画质 MuJoCo 场景（纹理/阴影/电影机位），双方块排序 |
| [videos/demo_omnijev_sort.mp4](videos/demo_omnijev_sort.mp4) | 基础场景 OmniJev 双方块排序成功 |
| [videos/demo_stepfun_sort.mp4](videos/demo_stepfun_sort.mp4) | StepFun VLM 对照组（答案抖动导致失败样本） |

## 架构 | Architecture

```
相机帧 ──> OmniJev (云端, 一次前向返回所有问题的概率)
              ├─ choice(region): 目标物体在哪个区域？→ 8x8 变焦细化
              ├─ noul: holding? done? grasp_now? release_now?
              └─ score: 任务进度 (just started ... almost done)
                   │
                   ▼
         PBVS 伺服：区域中心反投影到桌面平面 → OSC 20Hz 位置控制
                   │
                   ▼
         抓取验证（shake test 物理校验）→ 失败则 refine（近距精定位）重试
```

关键工程细节：

- **反投影定深**：模型指出的图像点沿相机射线投射到桌面固定平面（`table_z`，场景标定常数），消除单目深度歧义
- **答案防抖**：区域答案做 EMA 平滑；任务完成后模型自然回答 `done`
- **状态机**：`POINT → DESCEND → GRASP_WAIT → VERIFY → (carry) POINT → DESCEND → RELEASE → FINISH`，带卡死看门狗
- **精修来源**：历史 HUD 的 `REFINE (close-range)` 使用仿真物体坐标；具体介入见源码与日志

## 实测能力边界 | Measured Capability Boundaries

历史调试中，0.8B 模型（`omnijev-0.8b-kk`）的区域指向出现厘米级误差，抓取需要仿真位姿修正。这些结果来自少量固定任务；真实感知、深度估计和真机控制能力仍需单独验证。

## 快速开始 | Quickstart

```bash
# Stage 1: 自建 MuJoCo 场景（单环境）
pip install mujoco imageio imageio-ffmpeg pillow numpy
MUJOCO_GL=osmesa python run_show.py --backend omnijev --scene pretty_scene.xml --beauty

# Stage 2: LIBERO（独立环境，需 mujoco 2.3.x）
conda create -n libero python=3.10 -y
pip install robosuite==1.4.0 mujoco==2.3.7 numpy==1.23.5 opencv-python easydict \
            gym==0.25.2 cloudpickle imageio imageio-ffmpeg pillow matplotlib \
            future termcolor bddl==1.0.1 torch
git clone https://github.com/Lifelong-Robot-Learning/LIBERO
pip install -e LIBERO --no-deps
MUJOCO_GL=osmesa python libero_point_loop.py --task-idx 0 --max-decisions 400
```

环境变量：

```bash
export OMNIJEV_BASE=http://...     # OmniJev systemone 端点
export OMNIJEV_KEY=...             # demo key 见 OmniJev 官方文档
export STEPFUN_API_KEY=...         # 仅对比实验需要（bench_models.py）
```

## 文件结构

```
sim.py / sim_scene.xml / pretty_scene.xml   自建 MuJoCo 场景（含程序化纹理）
run_show.py                                 Stage-1 闭环演示（OmniJev/StepFun 双后端）
bench_models.py                             OmniJev vs StepFun 定量对比
omnijev_api.py                              OmniJev systemone API 客户端
libero_point_loop.py                        LIBERO 闭环主程序（region 指向 + PBVS + refine）
libero_loop.py                              早期版本（官方 direction/gripper 模板直接控制）
jev_benchmark/                              三组实验、API 客户端、三栏视频生成
tests/test_benchmark.py                     接口、动作放行与配对契约检查
docs/comparison.md                         实验分工、复跑方式与限制
probes/                                     所有探针/调试脚本（保留完整研发痕迹）
results/                                    历次实验日志 JSON
videos/                                     验证视频
```

## 致谢

- [OmniJev](https://github.com/tinnel123666888/OmniJev) — System-One 决策模型与 API
- [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) — 仿真基准与场景
- [robosuite](https://github.com/ARISE-Initiative/robosuite) — 机器人仿真框架
