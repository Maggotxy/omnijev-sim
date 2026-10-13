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
