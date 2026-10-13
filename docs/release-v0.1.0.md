# OmniJev 条件反射仿真 v0.1.0

下载入口：[GitHub Release v0.1.0](https://github.com/Maggotxy/omnijev-sim/releases/tag/v0.1.0)。现有 OmniJev 负责局部条件判断和动作，Gemini 3.8 Flash 作为异步视觉任务大脑。模型权重固定；随机物理、相机、小脑、大脑和录像分别持续运行。

## 下载内容

- `omnijev-sim-v0.1.0-runtime.zip`：连续仿真源码、Panda 机器人资源、依赖清单、复跑与分析脚本、25 项运行时测试、使用文档及许可证。
- `omnijev-sim-v0.1.0-evidence.zip`：最新 12 轮两组配对实验、六段对比视频、原始模型请求与图像、轨迹、执行源码快照，以及最终交互操作证据。
- `SHA256SUMS.txt`：两个下载包的 SHA-256 校验值。
- GitHub 自动提供的 Source code 下载：完整仓库，包括更早的抓放、压力测试和串行实验记录。

两个 ZIP 使用同一顶层目录 `omnijev-sim-v0.1.0/`，解压到同一位置即可合并。运行包可以单独启动；报告中的原始实验文件在证据包或完整仓库中。

## 独立启动

已验证环境为 Linux、Python 3.10、MuJoCo 2.3.7、CPU/OSMesa。Windows 用户可在 WSL 的 Linux 环境安装；本次没有验证原生 Windows MuJoCo。不需要 GPU 或本地模型权重，模型调用需要用户自行提供有权使用的接口和凭证。

Ubuntu/Debian 安装 OSMesa 系统库后，在解压目录执行：

```bash
sudo apt-get install -y libosmesa6
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

python -m jev_reflex.live --out out/reflex_interactive --port 8790
```

`OMNIJEV_BASE` 是支持 `/v1/systemone` 的 API 根地址。上述占位值需要替换为自己的地址和密钥；下载包不提供共享凭证，也不包含第三方服务额度。g.92.run 的模型名来自本次实际请求，未来可用性以服务端为准。

浏览器打开 <http://127.0.0.1:8790/>，选择组别并启动。可以输入自然语言条件、指定动作和阈值，加入随机人手、自由物体、移动目标和遮挡，查看条件概率与真实电机命令。纯 `omni` 组不请求视觉大脑；`vlm_omni` 组需要同时配置两个接口。停止后，记录写入 `out/reflex_interactive`。

## 检查与复跑

```bash
PANDA_ASSET_ROOT="$PWD/resources/panda" python -m unittest discover -s tests -p 'test_reflex*.py' -v
python scripts/run-serial-reflex.py --out out/paired_trial --seeds 313 337 --seconds 40 --scenarios quiet hand combined
python -m jev_reflex.compose out/paired_trial
python scripts/analyze-runtime.py out/paired_trial
```

这会调用已配置的模型接口。相同 seed 配对外部事件，但网络延迟和模型输出可能变化，因此结果不保证逐帧一致。测试里的固定答复只核验机制，不作为模型能力证据。

## 本次结果

最新 12 轮中，视觉大脑＋OmniJev 完成目标 4/6 次，纯 OmniJev 完成 2/6 次；两组各记录到 1/20 个类别匹配的新命令响应。采集到电机命令的中位延迟约 0.89 秒，大脑推理中位约 3.30 秒。详细口径和失败见[实验报告](reflex-runtime-results-20261013.md)。

简化刚体尚未覆盖柔性组织或全部生活情境。本版本提供可操作的仿真及可检查的证据，当前结果未证明可靠的通用条件反射能力或生物等价性。

## 许可与来源

项目代码沿用根目录 [MIT License](../LICENSE)。Panda 描述和网格遵循其 [Apache-2.0 License](../resources/panda/LICENSE)，来源和具体修改见 [SOURCE.md](../resources/panda/SOURCE.md)。运行包保留两个许可证和机器人资源哈希清单；不包含模型权重或用户提供的原始 PDF。
