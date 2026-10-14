# 部署记录

## 2026-10-14 已验证案例比较

- 继续在 bxj `/opt/LIBERO` 的实际 Python/MuJoCo 环境修改和验证。修改前备份 `out/source_before_qualified_cases_20261014`；未改变原 LIBERO 基线。
- 共享控制器按答复身份执行单次 1.5 厘米左右微调、总偏移限制 8 厘米；视觉绝对计划清除局部偏移，模型计划点附近保持。大脑组从机器人自身初始位置等待视觉计划。
- 增加连续低速保持到结束的任务完成评分与纯大脑初筛／冻结配对驱动。39 项机制检查通过，真实 API 初筛独立保存于 `out/qualified_screen_20261014` 和 `out/qualified_screen_extra_20261014`。
- 五个候选各运行三次，三个案例全部通过入选，两个案例各 2/3 未入选。选择清单为 `out/qualified_case_selection_20261014.json`，包含资格文件、运行源码和驱动哈希；正式配对为 `out/qualified_paired_20261014`。
- 源码最小回传归档 21,093 字节，两端 SHA-256 为 `ca502003158c71b270582549bd530a8f791fc46e618b639d18e73eb1c908787d`。运行凭证只在已有本任务进程与新实验进程的环境中传递。
- 本机 registry 无 JEV 匹配项，Cognee 工具不可用；沿用已核实的 SSH 别名和路径，记录存于本文件。
- 正式 18 轮完成，稳定成功为纯组 7/9、组合组 8/9；九段配对视频均 721 帧、20 fps、1920×800。运行源码与筛选一致，驱动和案例资格通过核验。
- 实验后修正交互重置的初始位置与最终图像字段，并显示稳定完成结果；控制与评分 AST 等价，差异与源码身份保存在 `results/qualified_ui_20261014/manifest.json`。最终 40 项机制检查通过。
- 新交互服务 PID 3456019，输出 `out/qualified_ui_20261014`，仅监听 loopback 8790；默认纯大脑、seed 701、36 秒。真实浏览器运行并刷新读回“稳定完成”，无 Omni 请求；结束后保持空闲。
- 最终源码与全部初筛、配对及 UI 原始记录归档 209,368,895 字节，本地与远程 SHA-256 均为 `1c85ded7ce40e8f3d43119b5c82f716bc5de97bf0d45b8d12f1dd044c4d32d84`。凭证精确扫描通过，原始图像与视频哈希匹配。

## 2026-10-14 视觉大脑与大脑＋小脑

- 在 bxj `/opt/LIBERO` 的现有运行环境直接修改 `jev_reflex` 和本任务的分析、实验及测试脚本；修改前备份为 `out/source_before_brain_comparison_20261014`，原 LIBERO 基线保留。
- 实验组为 `vlm` 与 `vlm_omni`，同用 `gemini-3.8-flash`，组合组额外请求 `omnijev-0.8b-kk`；凭证只通过环境传入。模型权重固定。
- 交互验收发现保持时参考点漂移，修正后重新执行完整 30 轮。最终目录为 `out/brain_cerebellum_hold_verified_20261014`：三个 seed、五类场景、每轮 26 秒，串行且按 seed 交换顺序；条件银行冻结。
- 35 项原生机制检查通过。十五段完整配对 MP4 各 521 帧、20 fps、1920×800；命令选择重放与全部 15,600 个实际控制 tick 一致，含 54 次缓存答复重新生效。
- 12 个运行模块在最终实验、交互服务与本地交付之间 SHA-256 一致；测试、源码快照、全部失败和分析输出保留。最终原始与媒体归档为 233,137,247 字节，传输两端 SHA-256 为 `1e4be7db51c74b57192153b67c04b11b27951fe68f240b7637103da5d7c4319a`。
- 当前服务 PID 3364689，仅监听 `127.0.0.1:8790`，输出 `out/brain_comparison_ui_release_20261014`。两组浏览器操作完成并停止保存，最终 `finished`；空闲不请求模型。本机通过隐藏 SSH 转发访问。
- 回传仅包含本任务源码和证据。修正前 30 轮、六轮试跑及旧交互证据独立保留，不合并进最终统计。
- 本机基础设施 registry 可读但无 JEV 对应项；沿用已核实的 SSH 别名和路径。Cognee 工具不可用，部署记录保存在本文件。

## 服务器架构总览

| 服务器 | 项目 | 远程路径 | 本地交付路径 |
|---|---|---|---|
| bxj（已有 SSH 别名） | LIBERO 仿真实验 | `/opt/LIBERO` | `omnijev-sim` |

## 2026-10-13 三组实验

- 核实服务器 hostname 为 bxj，LIBERO 基线提交为 `8f1084e3132a39270c3a13ebe37270a43ece2a01`。
- 现有 `libero_point_loop.py` 本地与远程 SHA256 一致；现有脚本保持原样。
- 新增 `/opt/LIBERO/jev_benchmark/`；测试脚本位于 `/opt/LIBERO/jev_benchmark_tests/`。
- 通过 SCP 传输新增文件，再在 `/opt/omnijev-sim/miniconda3/envs/libero/bin/python` 下编译、测试和运行。
- 大型内联写入命令被命令策略拒绝，因此使用可审阅补丁暂存新增文件，再以最小文件集传输到服务器验证。
- LIBERO 自带 NumPy 初始状态文件按官方任务路径载入，显式兼容当前 PyTorch 的 `weights_only` 默认值变更。
- 实验目录 `out/comparison_20261013*` 保留各次调试记录；正式交付批次另在报告中列出。
- 没有重启现有服务，没有安装本地模型。
- 部署记忆工具 Cognee 在当前会话不可用，记录保存在本文件。

## 2026-10-13 可用模型完整复跑

- 按后续指示使用 `gemini-2.5-flash`，完成 `gemini`、`gemini_omni`、`omni` 三组，seeds 为 1 和 7，官方初始状态索引为 0。
- 运行环境为 `/opt/omnijev-sim/miniconda3/envs/libero/bin/python`；模型凭证仅通过运行环境传入，未写入交付文件。
- 执行：`python -m jev_benchmark.run_suite --out out/comparison_gemini25_20261013 --seeds 1 7`。
- 合成：`python -m jev_benchmark.compose out/comparison_gemini25_20261013 --seeds 1 7`。
- 六轮全部执行完成，无 API 错误；两轮混合组成功，其余四轮失败。逐轮数值见 [结果报告](results-20261013.md)。
- 11 项测试通过；独立脚本控制器验收成功；执行源码哈希与本地交付源码一致。
- 最终目录 `/opt/LIBERO/out/comparison_gemini25_20261013` 通过 SCP 回传至 `results/comparison_gemini25_20261013`，含两段三栏视频和全部原始日志。
- 归档 21,747,689 字节，远程与本地 SHA256 均为 `516cb0289a8e185f91694f3ff1ac27d28cec7de2c09fe7ea5448fcc29e2390cf`；已安全解包并检查视频首、中、末帧。

## 2026-10-13 连续反射运行时

- 部署范围为 bxj 的 /opt/LIBERO/jev_reflex、scripts、tests 和 resources/panda；已有 LIBERO 基线保留。
- 使用现有 Python 3.10.22、MuJoCo 2.3.7 与 OSMesa；67 个 Panda 网格本地/远程 SHA256 已核验，原始碰撞与惯性保留。
- 两组为 omni / vlm_omni；实际接口标识为 omnijev-0.8b-kk 与 g.92.run gemini-3.8-flash。凭证仅传入进程环境。
- 已保存早期 24 轮控制比较、6 轮客户端试跑、24 轮压力数据、24 轮串行数据及最新 12 轮演员连续性修正后的配对数据；各执行版本有源码快照，不合并不同物理协议的结果。
- 最新命令：python scripts/run-serial-reflex.py --out out/reflex_runtime_verified_20261013 --seeds 313 337 --scenarios quiet hand combined --seconds 40。单进程串行，按 seed 交换组别顺序。
- 当前源码为固定模型权重的自然语言条件银行、模型仲裁、危险评分、反馈检索和异步视觉规划。发布评分器另过滤本轮结束之后的计划事件；与已执行快照的差异在 source_verification.json 中列明。
- 25 项机械/时序/契约测试通过；18 段新配对视频的实际帧数、20 fps、1920×800 均逐一核验；回传归档本地与远程 SHA256 一致。
- 最新原始归档 68,679,456 字节，SHA256 af9e184e7d98df3d53fdf5fbbd13270a09f8f37ee0b3afe864bcd69919db4f7a；成对媒体归档 68,115,066 字节，SHA256 70bcfda31afb4316d88d08de7e75f656a791ddd3acf31b9a17a6baedbe2d4002。
- 最终 UI 原始归档 60,067,874 字节，SHA256 9a725448e087875b2f840948c0e4c0177067dc67c2c1fbfc56a123feca1e4fae。真实浏览器验证了新增条件进入实际 API 问题、概率显示、两组切换和手动停止；验收回执只评价这些行为。
- 服务仅监听 127.0.0.1:8790，最终 PID 2730604，输出 out/reflex_interactive_ready_20261013；本机已有隐藏 SSH 转发，空闲 ready，不请求模型。
- 接触、误判、未匹配响应、过期保护、API 错误与任务失败均保留，结果见 reflex-runtime-results-20261013.md。
