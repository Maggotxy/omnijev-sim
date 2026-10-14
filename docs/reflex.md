# 连续条件反射仿真

Gemini 3.8 Flash 作为视觉大脑，独立或与 OmniJev 局部决策小脑组合。两组共享动作能力和视觉大脑提示词模板，模型权重固定。运行时可以用自然语言管理条件、保存动作与反馈并检索经验；物理世界在推理等待时持续运动。

## 两组实验

| 组别 | 任务意图 | 局部动作 |
|---|---|---|
| `vlm` | Gemini 3.8 Flash 看图更新目标，并选择有界动作 | 同一个视觉大脑 |
| `vlm_omni` | 相同视觉大脑与动作能力 | 额外使用独立 OmniJev 小脑 |

大脑返回归一化图像点，固定相机标定把这个**模型选择的点**投射到桌面。控制器不读取绿色目标真值来追踪目标。两组共享初始状态、环境随机事件和电机原语；真实轨迹会因动作不同而分开。[当前实验设计](brain-comparison-design.md)定义动作仲裁和速度口径。

## 实时交互

网页仅监听 `127.0.0.1:8790`。按照下方命令启动后打开 [实时仿真](http://127.0.0.1:8790/)。如果运行在远程主机，使用 `ssh -N -L 8790:127.0.0.1:8790 YOUR_SSH_HOST` 转发端口。空闲时不运行物理或请求模型。

1. 选择纯视觉大脑或视觉大脑＋OmniJev，点击**启动仿真**。
2. 随时伸入人手、引入黄色滑动物体、挪动绿色目标或遮挡相机。**切换随机干扰**持续产生随机间隔的变化；不同类型可以重叠。
3. 写一句条件，选择动作及概率阈值，点击**加入反射**。它立即变成新的 `noul` 问题，并出现在模型仲裁候选中；条件银行版本可见。
4. 观察条件概率条、阈值、模型候选、门控结果、实际电机动作与来源、实测速度、触觉和任务误差。危险评分是模型输出，不能直接当作真实风险概率。纯大脑组不请求小脑，界面的条件条没有小脑答复时显示空值。
5. 组合组可以**关闭小脑反射**，由视觉大脑继续选择动作；**开启小脑反射**后恢复局部判断。纯大脑组的小脑开关禁用。**重置场景**使在途旧回复失效，保留有世代标记的已完成经验。
6. **停止并保存**结束本轮并保存全部轨迹与模型记录；每轮最长 5 分钟。

左侧为观察相机，右侧为模型实际相机。橙圈来自机器人自身末端，蓝圈来自当前运动意图；世界事件及目标误差只供观察和评价。环境干扰的按钮名、时间表、真实物体坐标不会发送给模型。

**WATCHDOG HOLD** 表示软件因观察过期而保持姿态，另行统计。相机年龄和观察年龄可核对等待时间。已有感知在环境变化时可能变旧，这是连续闭环的一部分；环境事件本身不会替模型下达刹停命令。

## 物理与运行时

| 部分 | 实现 |
|---|---|
| 物理 | MuJoCo 2.3.7，500 Hz，Panda 原碰撞网格、惯性与执行器 |
| 电机控制 | 20 Hz，机器人自身关节和雅可比；速度上限 0.065 m/s |
| 人手 | 简化前臂、掌部和四指自由刚体，通过有限刚度约束接近、停留、离开，发生真实接触 |
| 黄色物体 | 外界在桌边引入独立自由刚体，旧物体继续运动；单轮最多 16 个，重置恢复容量；质量、摩擦、方向和速度随机 |
| 环境 | 手的高度、位置和进入速度，移动目标、部分或完整遮挡、照明、像素噪声随机 |
| 组合事件 | 九次随机间隔事件，覆盖四类并允许重复和异类重叠；同一个物理演员的两次事件不重叠 |
| 感知 | 独立线程，目标 8 Hz，保留采集时间，只处理最新帧 |
| 小脑 | 实际服务 `omnijev-0.8b-kk`；条件 `noul`、候选 `choice`、危险等级 `score` 一次批量返回 |
| 大脑 | g.92.run 原生 Gemini 接口，`gemini-3.8-flash`，low thinking，直接选择动作并更新任务目标；完成一轮后处理最新观察 |
| 命令有效期 | 小脑接收答复时采集年龄上限 1.2 秒，可执行至采集后 1.4 秒；大脑分别为 8 秒和 10 秒；实际采用的命令来源单独记录 |
| 动作原语 | 继续、刹停、局部退让约 6 cm 并抬高 4 cm、抬起约 8 cm、左右修正；关节限位分别记录 |

条件概率超过该条件的阈值且模型仲裁置信度超过 0.12，才执行选中的反射。否则继续当前任务意图；危险评分连续调节速度。条件与候选共享同一帧、同一次批量请求，候选不是读取本轮返回概率后进行第二次推理。

二元条件保持简短。检索经验进入动作仲裁和视觉大脑；每条经验保存开始/结束触力、实际位移、期间发生的动作和世代。它们是测量关联，不能把 0.6 秒内的反馈归因到某个单独动作。大脑只能修改已有条件或阈值，操作者可以新增条件；所有修改有来源与版本。

小脑、大脑、相机、录像使用独立线程。录像按真实墙钟 20 fps 采样，记录模型等待期间的持续物理。经典控制器、有界动作和过期保护分别报告。

## 复跑

Linux / Python 3.10 使用以下命令；需要 OSMesa 系统库。Panda 资源已经包含在仓库中。

```bash
pip install -r requirements-reflex.txt
export MUJOCO_GL=osmesa
export OMNIJEV_BASE=your_authorized_systemone_base_url
export OMNIJEV_MODEL=omnijev-0.8b-kk
export OMNIJEV_KEY=your_authorized_key
export GEMINI_BASE=https://g.92.run
export GEMINI_MODEL=gemini-3.8-flash
export GEMINI_API_KEY=your_authorized_relay_key

PANDA_ASSET_ROOT="$PWD/resources/panda" python -m unittest discover -s tests -p 'test_reflex*.py' -v

# 串行执行，按 seed 交换组别顺序；交互页面保持空闲
python scripts/run-serial-reflex.py --out out/reflex_runtime --seeds 541 577 613 --seconds 26 --scenarios quiet hand_close slider_fast occlusion_brief target_jump
python scripts/analyze-runtime.py out/reflex_runtime
python scripts/analyze-brain-motion.py out/reflex_runtime
python scripts/replay-command-selections.py out/reflex_runtime
python scripts/compose-brain-comparison.py out/reflex_runtime

# 独立交互使用
python -m jev_reflex.live --out out/reflex_interactive --port 8790
```

## 判定与证据

[实验设计](brain-comparison-design.md)定义假设与感知边界。首条匹配新命令必须来自事件开始之后的新鲜观察，类别符合预设、动作发生转换并真正改变电机命令。重复保持和提前退让不算新的事件响应；机械实际速度独立保存。重叠事件可能共享一个响应，不能据此断言模型识别了哪种事件。

各轮保存场景 XML、随机参数、控制轨迹、模型可见原始答复、发送 JPEG 及哈希、条件版本、检索反馈、大脑目标更新与真实墙钟录像。分析器核对配对、源码及输入图像身份。无干扰人手条件的负例概率单独报告；这种负例检查不代表包含正例的完整校准。

- [早期 24 轮云端反射结果](reflex-results-20261013.md)，保留原执行源码快照。
- [最新大脑／大脑＋小脑配对报告](brain-comparison-results-20261014.md)与[最终交互验收](../results/brain_comparison_ui_release_20261014/qa_receipt.json)。
- [上一轮小脑／大脑＋小脑配对报告](reflex-runtime-results-20261013.md)。
- [资源来源与许可证](../resources/panda/SOURCE.md)。

模型是否可靠由真实结果判断。简化刚体没有覆盖柔性组织、全部生活情境或真机；低接触力、合法输出、一次成功录像均不足以证明通用条件反射理论。
