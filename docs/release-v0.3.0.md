# v0.3.0：已验证案例上的大脑与小脑对比

本版先用纯视觉大脑重复筛选可完成的案例，再进行独立的两组配对。两组均从视觉模型选定目标后开始到达，使用同一套有界纠偏与保持控制器。模型权重固定。

## 下载与启动

[GitHub v0.3.0](https://github.com/Maggotxy/omnijev-sim/releases/tag/v0.3.0) 提供运行包、当前实验数据包与 SHA-256。两个 ZIP 共用 `omnijev-sim-v0.3.0/` 顶层目录。

已验证 Linux、Python 3.10、MuJoCo 2.3.7 与 OSMesa。安装 OSMesa 系统库后：

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
python -m jev_reflex.live --out out/interactive --port 8790 --seed 701
```

访问 <http://127.0.0.1:8790/>。纯大脑组只需要视觉接口；组合组额外需要 OmniJev。模型和外界状态存在随机性，既往通过不保证新运行也通过。

## 重复筛选和正式比较

```bash
PANDA_ASSET_ROOT="$PWD/resources/panda" python -m unittest discover -s tests -p 'test_reflex*.py' -v
python scripts/run-qualified-comparison.py --phase screen --out out/screen1 --seconds 36 --repeats 3 --cases quiet:701 occlusion_brief:709 target_jump:719
python scripts/run-qualified-comparison.py --phase screen --out out/screen2 --seconds 36 --repeats 3 --cases hand_close:751 occlusion_brief:743
python scripts/merge-qualified-cases.py --inputs out/screen1 out/screen2 --out out/selection.json
python scripts/run-qualified-comparison.py --phase paired --out out/paired --repeats 3 --qualification out/selection.json
python scripts/analyze-qualified-cases.py out/paired
python scripts/compose-qualified-cases.py out/paired
pip install -r requirements-analysis.txt
python scripts/plot-qualified-cases.py out/paired
```

每个目录必须新建。正式比较使用同一批已合格案例、相同源代码与模型策略，重新运行两个组；分析阶段校验驱动、运行源码、模型策略与重复次数。全部失败留在分母中，只有两组都成功的配对才计算完成时间差。

成功要求真实目标误差与实测速度满足条件，连续保持至少 40 个 20 Hz 控制步，并持续至本轮结束；事件后恢复单独计时。完整标准见[实验设计](qualified-cases-design.md)，结果见[实测报告](qualified-cases-results-20261014.md)。

项目代码为 MIT，Panda 资源保留 Apache-2.0 与来源说明。证据包保留执行源码快照、全部初筛、配对记录及一倍速视频。该有限仿真评估的是已筛选情境的任务结果。
