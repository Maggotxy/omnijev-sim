# 仿真服务运维记录

## 2026-10-13 仿真服务核验

- 服务器：已有 bxj SSH 别名，项目 /opt/LIBERO。
- 只读诊断：ss -tlnp；ps 按 CPU 排序；API 模型目录状态检查（未重启推理提供方或其他项目服务）。
- 实验并行压力出现超时，所有记录保留；正式比较采用串行执行。长上下文与短条件同图探针均留存，不能把两种变化的影响混为单一因果结论。
- 仅重启本任务创建的 jev_reflex.live，重启前检查 /state 非 running/loading 且 /proc/PID/cmdline 身份匹配。
- 最终服务 PID 2730604，监听 127.0.0.1:8790；读回 status=ready、selected_group=vlm_omni。空闲没有 API 调用。
- 发布源哈希与本地代码一致；25 项检查通过，UI 操作与持久化有独立回执。
- infrastructure-registry 脚本在本机不可用；沿用已核实的 SSH 别名和项目部署记录。Cognee 工具在此会话不可用，运维与部署记录保存在仓库文档。
