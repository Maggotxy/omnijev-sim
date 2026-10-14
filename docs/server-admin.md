# 仿真服务运维记录

## 2026-10-14 两组比较与最终交互服务

- 继续使用 bxj `/opt/LIBERO` 和既有 Python 3.10.22 / MuJoCo 2.3.7 / OSMesa；没有重启推理提供方或其他项目服务。
- 只重启本任务创建的 `jev_reflex.live`，重启前确认任务已停止并核对进程身份。最终 PID 3364689，监听 `127.0.0.1:8790`。
- 当前输出 `out/brain_comparison_ui_release_20261014`；纯大脑和组合组各一次浏览器验收已停止保存，读回状态 `finished`、`selected_group=vlm_omni`、条件银行版本 1。纯组无 Omni 请求；组合组新增条件进入真实请求并收到概率。
- 35 项机制检查通过，包含刹停和软件保持各 8 秒、位移小于 5 mm 的回归约束。实测性能与失败见 `brain-comparison-results-20261014.md`；UI 回执不评价模型能力。
- 本机隐藏 SSH 转发由 `scripts/open-reflex.ps1` 管理；断开后可重建，不改变远端 API 配置。停止状态没有新的物理运行或模型调用。
- 本机 registry 无 JEV 对应项目，Cognee 工具不可用；服务位置、源码身份与操作依据记录在本文件及 `deploy-tracker.md`。

## 2026-10-13 仿真服务核验

- 服务器：已有 bxj SSH 别名，项目 /opt/LIBERO。
- 只读诊断：ss -tlnp；ps 按 CPU 排序；API 模型目录状态检查（未重启推理提供方或其他项目服务）。
- 实验并行压力出现超时，所有记录保留；正式比较采用串行执行。长上下文与短条件同图探针均留存，不能把两种变化的影响混为单一因果结论。
- 仅重启本任务创建的 jev_reflex.live，重启前检查 /state 非 running/loading 且 /proc/PID/cmdline 身份匹配。
- 最终服务 PID 2730604，监听 127.0.0.1:8790；读回 status=ready、selected_group=vlm_omni。空闲没有 API 调用。
- 发布源哈希与本地代码一致；25 项检查通过，UI 操作与持久化有独立回执。
- infrastructure-registry 脚本在本机不可用；沿用已核实的 SSH 别名和项目部署记录。Cognee 工具在此会话不可用，运维与部署记录保存在仓库文档。
