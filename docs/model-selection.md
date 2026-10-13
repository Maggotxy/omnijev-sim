# 视觉大脑接口选择：2026-10-13

当前使用用户提供的 g.92.run 接口，默认模型是 **Gemini 3.8 Flash**。它已经通过真实图片请求，并在持续仿真中返回目标图像点、等待视野恢复的计划，以及条件/阈值更新。鉴权只通过运行环境传入。

| 接口模型 ID | 实际图片请求 | 观察 |
|---|---|---|
| `gemini-3.8-flash` | 成功，响应 `modelVersion` 同名 | 初次图片约 7.7 秒；明确任务的手部/遮挡检查约 7.3/7.5 秒；实现客户端的一次完整检查约 9.0 秒 |
| `gemini-3.5-flash-lite` | 成功，响应 `modelVersion` 同名 | 初次图片约 2.7 秒，可作为速度优先的备用；尚未完成相同规模的闭环比较 |
| `gemini-3.7-flash` | 目录列出，但实际请求 404 | 当前不可据目录存在判断可用 |

3.8 的任务提示明确：绿色/青绿色圆垫是目标，黄色物体是障碍，夹爪保持打开。它只更新较慢的任务意图，OmniJev 同时根据当前相机与触觉执行局部反射。网络耗时保留在物理世界和录像中。

客户端使用原生 `generateContent`、JSON schema 和 `thinkingLevel=low`；该设置属于 [Google 当前 Gemini 文档](https://ai.google.dev/gemini-api/docs/latest-model?hl=en)描述的支持级别。实际响应标识由中转接口提供，本文记录接口事实。

模型目录、单张图片调用、闭环任务效果分别核验。一次正确回答不足以证明整体能力；运行中的错误、延迟、目标更新和条件更新都进入实验日志。OmniJev 的 `noul`、`choice`、`score` 接口依据 [官方仓库](https://github.com/tinnel123666888/OmniJev)；当前实际接入的服务标识仍为 `omnijev-0.8b-kk`，不把官方其他版本的成绩套用到该服务。

证据：[初次图片探针](../results/g92_vision_probe_20261013.json)、[手部与遮挡探针](../results/g92_brain_probe_20261013.json)、[实现客户端检查](../results/g92_runtime_client_probe_20261013.json)。
