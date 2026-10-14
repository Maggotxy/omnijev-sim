# v0.2.0：大脑与大脑＋小脑的速度对比

主比较为 `vlm` 与 `vlm_omni`。两组使用同一个视觉大脑、相同动作能力和外界事件；组合组额外启用独立 OmniJev 局部判断。模型权重固定，物理世界持续运行。

## 下载

[GitHub Release v0.2.0](https://github.com/Maggotxy/omnijev-sim/releases/tag/v0.2.0) 提供运行 ZIP、当前实验与交互证据 ZIP，以及 SHA-256 校验文件。运行包包含 Panda 资源、源码、依赖、测试和说明；证据包包含原始请求与图像、轨迹、执行源码快照、十五段完整配对视频和分析图。

两个包使用同一顶层目录 `omnijev-sim-v0.2.0/`，可以解压到同一位置。GitHub 的完整源码下载保留此前各版实验与修正前记录。

## 启动

已验证 Linux / Python 3.10 / MuJoCo 2.3.7 / OSMesa。准备 Python 3.10 和 OSMesa 系统库，在解压目录执行：

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements-reflex.txt
export MUJOCO_GL=osmesa
export OMNIJEV_BASE=https://YOUR_SYSTEMONE_API_HOST
export OMNIJEV_MODEL=omnijev-0.8b-kk
export OMNIJEV_KEY=YOUR_AUTHORIZED_KEY
export GEMINI_BASE=https://g.92.run
export GEMINI_MODEL=gemini-3.8-flash
export GEMINI_API_KEY=YOUR_AUTHORIZED_RELAY_KEY
python -m jev_reflex.live --out out/interactive --port 8790
```

浏览器访问 <http://127.0.0.1:8790/>。选择纯视觉大脑或组合组后启动，可以加入近手、横穿、短暂遮挡和目标移动，查看命令来源及实测速度。纯大脑组只使用视觉大脑凭证；组合组同时需要 OmniJev 接口。接口地址和密钥由使用者自行配置。

## 复跑与查看运动

```bash
PANDA_ASSET_ROOT="$PWD/resources/panda" python -m unittest discover -s tests -p 'test_reflex*.py' -v
python scripts/run-serial-reflex.py --out out/comparison --seeds 541 577 613 --seconds 26 --scenarios quiet hand_close slider_fast occlusion_brief target_jump
python scripts/analyze-runtime.py out/comparison
python scripts/analyze-brain-motion.py out/comparison
python scripts/replay-command-selections.py out/comparison
python scripts/compose-brain-comparison.py out/comparison
pip install -r requirements-analysis.txt
python scripts/plot-brain-comparison.py out/comparison
```

视频按真实墙钟 20 fps、1 倍速度回放。命令年龄、动作变化、实际运动、任务计划更新和软件保持分别统计；比较方式见 [实验设计](brain-comparison-design.md)。相同 seed 保证配对环境参数，网络延迟和模型输出仍可能变化。

本版修正持续保持时关节参考点随重力下沉的问题，刹停和软件保持冻结已有参考点。35 项机制检查通过，包含两种保持路径在无外部接触下持续 8 秒、末端位移小于 5 mm 的检查。测试替身只核验机制；模型是否及时且正确地行动由真实接口记录判断。

项目代码沿用 [MIT License](../LICENSE)，Panda 资源保留 [Apache-2.0 License](../resources/panda/LICENSE) 与[来源说明](../resources/panda/SOURCE.md)。这是有限刚体仿真和有界动作系统，当前实验不能证明全部生活情境中的能力或生物等价性。
