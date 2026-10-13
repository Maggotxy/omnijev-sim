# OmniJev Closed-Loop Robot Simulation

真闭环机器人仿真：云端 [OmniJev](https://github.com/tinnel123666888/OmniJev) System-One 决策模型直接控制 MuJoCo / LIBERO 仿真中的 Franka Panda 机械臂完成多物体操作任务。

> **Unofficial research demo.** OmniJev 官方发布的机器人 demo 是 LIBERO 专家轨迹的离线回放（README 原文：*"The released demonstrations are offline replays; reliable closed-loop gameplay and robot control have not been established"*）。本仓库实现的是**真正的闭环控制**：每个决策都来自模型对当前画面的实时推理。

## 亮点 | Highlights

- 🎯 **LIBERO-10 官方任务真闭环成功**："put both the alphabet soup and the tomato sauce in the basket" — 98 次决策、39.6 秒仿真时间、环境官方 `success=True` 判定
- 🧠 **模型在训练分布内**：LIBERO 正是 OmniJev `robot_long` 训练族的场景来源；使用官方 v1.1 region-box 指向格式（`options` 带 `box` 坐标）和官方 LIBERO 问题模板
- 👁️ **语义/几何分层架构**：模型负责语义（选物体、指区域、进度、完成判定），近距精修（refine，等价腕部相机/深度传感器）补最后几厘米
- 📊 **OmniJev vs StepFun VLM 定量对比**：格子定位 11/16 vs 13/16、决策 3/7 vs 4/7、延迟 0.53s vs 0.90s、闭环成功率 2/2 vs 0/2
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
- **诚实标注**：HUD 中 `REFINE (close-range)` 明确标出精修介入的时刻

## 实测能力边界 | Measured Capability Boundaries

0.8B 模型（`omnijev-0.8b-kk`，官方共享 demo 端点）在 LIBERO 杂乱场景下的裸指向精度约 **5-10cm**——足以选中正确的物体和区域，但不足以直接合拢抓取（罐体半径 ~3cm）。这与官方 "closed-loop not established" 的声明一致。refine 机制（等价于真实机器人的腕部相机近距标定）补齐了这一差距。

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
probes/                                     所有探针/调试脚本（保留完整研发痕迹）
results/                                    历次实验日志 JSON
videos/                                     验证视频
```

## 致谢

- [OmniJev](https://github.com/tinnel123666888/OmniJev) — System-One 决策模型与 API
- [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) — 仿真基准与场景
- [robosuite](https://github.com/ARISE-Initiative/robosuite) — 机器人仿真框架
