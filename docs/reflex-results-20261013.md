# 连续反射实验：2026-10-13

完成 **24 轮、12 组成对试验**。每轮 20 秒连续物理，seeds 73、97 未用于前期提示调试。六类情况为无干扰、人手、滑动物体、目标移动、遮挡及随机顺序组合。

符合预设类别且真正改变电机命令的事件后响应为 **1/16**。原计划对照组最终到达目标 **8/12**；模型反射组 **0/12**。到达目标与避让效果必须分别看待。

## 时序与接口

- 世界实时因子 0.999988–0.999993，物理未等待模型。
- 决策调用完成 335 次，接受 295 次，错误 12 次；结束后返回的答复及过期答复不应用。
- 推理耗时 p50 **651 ms**，p95 **1540 ms**；包括超时请求。
- 已接受答复的采集到命令应用延迟 p50 **696 ms**，p95 **1046 ms**。
- 模型组平均 **36.0%** 的控制周期处于软件过期保护。它不属于模型反射。
- 模型 `omnijev-0.8b-kk`；执行方模型标识：`omnijev-0.8b-kk`。本地标识由本地执行器记录，云端标识来自服务端头。

## 全部配对结果

人手/物体接触列为整轮峰值，单位 N；左右分别为原计划对照/模型反射。干扰前干预是无干扰前缀中非继续、非 watchdog 命令的周期比例，quiet 使用整轮。

| 情况 / seed | 最终到目标 对照/模型 | 目标误差 cm 对照/模型 | 人手峰值 对照/模型 | 物体峰值 对照/模型 | 事件后响应 | 干扰前干预 | watchdog | 视频 |
|---|---|---|---|---|---|---|---|---|
| combined / 73 | 否/否 | 11.7/10.2 | 202.1/209.9 | 0.0/0.0 | 0/4 | 55.7% | 41.8% | [成对录像](../results/reflex_validation_20261013/comparison_combined_seed73.mp4) |
| combined / 97 | 否/否 | 11.5/11.5 | 204.2/158.5 | 0.0/0.0 | 0/4 | 0.0% | 74.8% | [成对录像](../results/reflex_validation_20261013/comparison_combined_seed97.mp4) |
| hand / 73 | 是/否 | 0.0/22.2 | 50.6/0.0 | 0.0/0.0 | 0/1 | 82.0% | 11.5% | [成对录像](../results/reflex_validation_20261013/comparison_hand_seed73.mp4) |
| hand / 97 | 是/否 | 0.0/4.2 | 147.1/191.2 | 0.0/0.0 | 1/1 | 51.5% | 43.2% | [成对录像](../results/reflex_validation_20261013/comparison_hand_seed97.mp4) |
| occlusion / 73 | 是/否 | 0.0/4.3 | 0.0/0.0 | 0.0/0.0 | 0/1 | 57.4% | 40.5% | [成对录像](../results/reflex_validation_20261013/comparison_occlusion_seed73.mp4) |
| occlusion / 97 | 是/否 | 0.0/7.1 | 0.0/0.0 | 0.0/0.0 | 0/1 | 48.5% | 49.2% | [成对录像](../results/reflex_validation_20261013/comparison_occlusion_seed97.mp4) |
| quiet / 73 | 是/否 | 0.0/27.8 | 0.0/0.0 | 0.0/0.0 | 0/0 | 97.0% | 0.2% | [成对录像](../results/reflex_validation_20261013/comparison_quiet_seed73.mp4) |
| quiet / 97 | 是/否 | 0.0/5.1 | 0.0/0.0 | 0.0/0.0 | 0/0 | 20.0% | 41.0% | [成对录像](../results/reflex_validation_20261013/comparison_quiet_seed97.mp4) |
| slider / 73 | 是/否 | 0.0/31.7 | 0.0/0.0 | 0.0/0.0 | 0/1 | 82.0% | 0.8% | [成对录像](../results/reflex_validation_20261013/comparison_slider_seed73.mp4) |
| slider / 97 | 是/否 | 0.0/7.1 | 0.0/0.0 | 0.0/0.0 | 0/1 | 25.8% | 50.2% | [成对录像](../results/reflex_validation_20261013/comparison_slider_seed97.mp4) |
| target / 73 | 否/否 | 14.9/32.0 | 0.0/0.0 | 0.0/0.0 | 0/1 | 82.0% | 0.2% | [成对录像](../results/reflex_validation_20261013/comparison_target_seed73.mp4) |
| target / 97 | 否/否 | 14.9/12.2 | 0.0/0.0 | 0.0/0.0 | 0/1 | 62.1% | 79.0% | [成对录像](../results/reflex_validation_20261013/comparison_target_seed97.mp4) |

## 每个事件的首个命令响应

| 情况 / seed | 事件 | 开始 s | 首动作 | 开始到命令 s | 采集到命令 ms |
|---|---|---|---|---|---|
| combined / 73 | hand | 3.091 | 未记录匹配响应 | — | — |
| combined / 73 | occlusion | 4.346 | 未记录匹配响应 | — | — |
| combined / 73 | slider | 5.836 | 未记录匹配响应 | — | — |
| combined / 73 | target | 8.265 | 未记录匹配响应 | — | — |
| combined / 97 | hand | 3.309 | 未记录匹配响应 | — | — |
| combined / 97 | target | 5.778 | 未记录匹配响应 | — | — |
| combined / 97 | occlusion | 7.061 | 未记录匹配响应 | — | — |
| combined / 97 | slider | 9.429 | 未记录匹配响应 | — | — |
| hand / 73 | hand | 3.091 | 未记录匹配响应 | — | — |
| hand / 97 | hand | 3.309 | retreat | 2.291 | 944 |
| occlusion / 73 | occlusion | 3.091 | 未记录匹配响应 | — | — |
| occlusion / 97 | occlusion | 3.309 | 未记录匹配响应 | — | — |
| slider / 73 | slider | 3.091 | 未记录匹配响应 | — | — |
| slider / 97 | slider | 3.309 | 未记录匹配响应 | — | — |
| target / 73 | target | 3.091 | 未记录匹配响应 | — | — |
| target / 97 | target | 3.309 | 未记录匹配响应 | — | — |

## 解释与限制

响应表使用事先定义的动作类别，只证明事件之后发生了匹配的命令转换。尤其在重叠干扰中，同一个命令可能匹配多个事件，不能据此判断模型识别了哪个事件。独立事件与组合事件分别列出。实际末端速度保存于 states.jsonl，可检查后续机械运动。

干扰前误退让可能提前让机器人远离接触区域，因此接触力下降本身不能证明正确条件反射。错误动作、持续保持、任务失败和 watchdog 均保留。当前云端模型的可靠性和延迟仍限制反射效果。

配对按同一 Scenario、初始状态哈希、原运动计划和软件限制核验；执行顺序固定为对照后模型，没有顺序平衡。试验是简化刚体场景的小样本，不能外推到全部日常情况、毫秒级人类反射或真人安全。

## 可复核材料

- [原始汇总](../results/reflex_validation_20261013/summary.json)、[机器可读分析](../results/reflex_validation_20261013/metrics.json)、[配对 CSV](../results/reflex_validation_20261013/pairs.csv)。
- [运行源码哈希和参数](../results/reflex_validation_20261013/manifest.json)、[机器人网格哈希](../results/reflex_validation_20261013/mesh_manifest.json)、[录像参数](../results/reflex_validation_20261013/videos.json)。
- 每轮目录包含完整 scene.xml、scenario.json、states.jsonl、applications.jsonl、api.jsonl，以及每次实际发送 JPEG；分析器核验模型记录的 JPEG 哈希。
- 机械/异步测试通过，测试中固定响应明确标为测试替身，不作为模型效果证据；本地权重输入契约另有测试。
- 浏览器交互另有 [验收回执](../results/reflex_ui_20261013/qa_receipt.json)；它核验按钮与持久化，不证明模型普遍正确。
- 使用 `python scripts/summarize-reflex.py results/reflex_validation_20261013 --report docs/reflex-results-20261013.md --title '连续反射实验：2026-10-13'` 可重新生成此报告。
