# Gemini 与 OmniJev 的受控闭环比较

实验任务为 LIBERO-10 task 0：把 alphabet soup 和 tomato sauce 放入 basket。三组使用相同的官方初始状态、随机种子、物体目录、仿真位姿辅助、20 Hz OSC 控制器和抓放技能。

| 组别 | 子任务规划 | 动作阶段放行 |
|---|---|---|
| `gemini` | Gemini 2.5 Flash | Gemini 2.5 Flash |
| `gemini_omni` | Gemini 2.5 Flash | OmniJev |
| `omni` | OmniJev | OmniJev |

规划器从场景物体中选择下一组抓取对象和放置目标。每个技能的 APPROACH、DESCEND、GRASP、LIFT、CARRY、LOWER、RELEASE 阶段都调用反射层，返回 proceed、wait 或 replan。wait 保持姿态，replan 中止本次尝试并调用共同恢复动作。OmniJev 对 proceed 的概率不足 0.5 时保持等待；最多等待四次后停止实验，不通过超时强制执行抓放。

Gemini+OmniJev 的分工是低频任务规划与动作阶段放行。这里的反射层在阶段边界调用，尚未实现并发的连续视觉监控。所有 API 请求等待期间仿真暂停，视频按仿真时间 1× 播放；真实耗时、API 等待时间单独统计。

## 感知和控制边界

三组都使用仿真器提供的物体标签和精确位姿。控制器将目标物体高度增加超过 5 cm 作为抓取成功的代理信号，通过相对位置记录放置反馈；高度信号并不直接测量夹爪实际夹持了哪件物体。它们是共享的仿真特权信息，不能据此推断真实深度相机、检测器或视觉模型的定位能力。模型负责选择任务与放行动作；关节动作由相同的经典控制器生成。

环境 `check_success()` 是最终任务判定，模型的 finish 和环境成功分别记录。API 不可用的组别记作 unavailable，成功字段为 null，不计作失败样本。当前控制器仅支持上述双物体入篮任务。

## 运行

在安装 LIBERO 的 Python 3.10 环境中，将本仓库加入 PYTHONPATH。实测依赖版本见 [requirements-libero.txt](../requirements-libero.txt)，它记录当前环境，尚未在全新机器上重建验证。LIBERO 版本为 `8f1084e3132a39270c3a13ebe37270a43ece2a01`。

```bash
export PYTHONPATH=/path/to/omnijev-sim:$PYTHONPATH
export MUJOCO_GL=osmesa
export GEMINI_BASE=https://g.92.run
export GEMINI_MODEL=gemini-2.5-flash
export GEMINI_API_KEY=YOUR_KEY
export OMNIJEV_BASE=http://YOUR_ENDPOINT
export OMNIJEV_KEY=YOUR_KEY
python -m jev_benchmark.run_suite --out out/comparison --seeds 1 7 --init-index 0
python -m jev_benchmark.compose out/comparison --seeds 1 7
python -m unittest discover -s tests -v
# 机械验收（脚本指定顺序，成绩不属于任何模型）
python -m jev_benchmark.check_controller --out out/controller_check
```

输出目录必须为新目录，以保留早期结果。默认两个 seed 都使用官方初始状态索引 0，属于同一布局的重复测量。跨布局实验应更改 `--init-index` 并使用新的输出目录。生成器产生 1536×816、20 fps 三栏视频，标题由实际模型名生成；无法执行的模型显示静态 NOT RUN 面板。

每个 episode 包含 `result.json`、`decisions.jsonl`、`actions.jsonl`、`initial.png`、`final.png` 和 `episode.mp4`。批次级 API JSONL 记录请求目的、请求及服务端模型名、图像哈希、完整响应和实测 HTTP 耗时。初始状态、静置后状态及最终状态都有 SHA256；视频合成检查已执行组别的初始状态和控制器是否一致。

## 模型与官方接口核对（2026-10-13）

当前实测模型为 `gemini-2.5-flash`。最初指定的 Robotics ER 2 接口不可用后，按可用模型继续实验。客户端通过 `GEMINI_MODEL` 选择模型并记录服务端返回的 `modelVersion`；不会在请求失败后静默替换模型。

- [Robotics overview](https://ai.google.dev/gemini-api/docs/robotics-overview) 列出 `gemini-robotics-er-2-preview` 与 `gemini-robotics-er-2-streaming-preview`。
- [Spatial reasoning](https://ai.google.dev/gemini-api/docs/robotics-spatial) 中的图像点为 `[y, x]`、0–1000 归一化坐标。本实验共享仿真位姿，不使用点坐标来执行抓取。
- [Task orchestration](https://ai.google.dev/gemini-api/docs/robotics-orchestration) 展示把机器人函数交给模型编排。当前实现将有限的规划和放行结果映射到共享技能。
- [Robotics with streaming](https://ai.google.dev/gemini-api/docs/robotics-streaming) 要求 Streaming 端点通过 Live API 使用；如设置 streaming 模型名，本 HTTP 客户端会明确拒绝错误的接法。
- 客户端支持 Gemini `generateContent` 和 `interactions` 两种协议，通过 `GEMINI_PROTOCOL` 选择。中转服务是否兼容由实际请求确认。

当天接口检查：g.92.run 的普通 Gemini 2.5 Flash 返回 HTTP 200；R2 generateContent 返回 HTTP 404，提示找不到模型或项目无访问权；Interactions 返回 Unsupported model interaction。Streaming 通过标准 Live API WebSocket 路径握手后，在模型初始化阶段收到 close code 1011，关闭原因提及 Publisher model。这个错误不能单独判定是模型权限、区域还是中转路由配置问题。未来评估 R2 时需先取得可调用接口。

## 统计解释

少量固定布局实跑用于验证控制链路和排查失效模式，不能支持某一模型整体优于另一模型的结论。旧版 StepFun 视频是不同控制管线的一次双子任务实验；其中 2/2 表示一个 episode 内两个子任务，不是两个独立 episode 的成功率。
