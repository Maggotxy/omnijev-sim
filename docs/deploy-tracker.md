# 部署记录

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
