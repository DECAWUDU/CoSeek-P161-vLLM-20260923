# CoSeek P161 · 内网 Qwen3.5-397B vLLM 部署包

本包将 **Planner、远端 Overview、frame_verify 和远端最终回答** 全部接到同一 OpenAI 兼容 vLLM 服务。本地 Qwen3.5-9B／4B 仍负责搜索与本地 Overview。已有 397B 服务不重装、不重启、不修改。

基于近期使用的 **P161 decoder-repaired＋I-frame／本地双路 Overview**，包含五选项和字幕路径。`runtime/` 与冻结源码逐字节一致；P162 仍在对照验证，未混入此次模型替换，避免同时改变算法和模型。

## 1. 最小部署

运行平台 Linux x86_64。Agent 可以放在能访问 `10.90.79.125:8077` 的另一台机器；视频文件和小模型权重只需对 Agent 机器可见。397B 接收请求中的图像数据，不需要访问视频路径。

```bash
unzip CoSeek-P161-vLLM-20260923.zip
cd CoSeek-P161-vLLM-20260923
bash setup.sh --main-only
```

`--main-only` 只在包目录创建主环境，适合先检查 API，或使用已有小模型环境。若需要同时安装小模型环境，运行 `bash setup.sh --full`；会另建 `.venvs/qwen`，使用 PyTorch 2.10.0 / torchvision 0.25.0 / CUDA 12.8 wheel 及随包的 Transformers wheel。需要相应驱动；不会安装或改变驱动、vLLM 包、397B 权重。

`setup.sh --download-model` 额外从 Hugging Face 下载 9B 权重。默认不下载权重。联网安装所需的系统工具为 ffmpeg、ffprobe、git；这不是离线依赖镜像。

编辑生成的 `.env`：

```bash
OPENAI_API_BASE='http://10.90.79.125:8077/v1'
OPENAI_API_KEY='EMPTY'       # 服务启用鉴权时换为真实密钥
OPENAI_MODEL='AUTO'         # 自动读取 /v1/models；多模型时必须写准确 ID
VLLM_ENABLE_THINKING='server'
VLLM_JSON_MODE='true'
COSEEK_GPU='5'              # 示例：必须改成实际供小模型使用的 GPU
QWEN_MODEL_PATH='/data/models/Qwen3.5-9B'
QWEN_PYTHON='/path/to/qwen-env/bin/python'
```

不要直接假定模型 ID 是 `Qwen/Qwen3.5-397B-A17B`：如果服务使用了 `--served-model-name`，应使用其返回的 ID。`AUTO` 只在恰有一个模型时自动选择。

四张 B200 已承载 397B，并不代表自动还有小模型运行空间。小模型 GPU 必须显式设置；可以将 Agent＋9B 放在另一张卡／另一台可访问服务的机器上。本包默认通过本机常驻进程运行小模型，不把搜索也转给 397B。4B 消融可改 `QWEN_MODEL_PATH` 为 4B 权重路径；其精度需单独统计。

## 2. 按顺序检查

```bash
./run.sh verify
./run.sh probe
```

`probe` 不加载 9B，不上传用户视频，只检查模型发现、简单文本、JSON 和 32 张合成图像的收图／顺序识别能力；每个请求最多 400 秒，SDK 不重试。若服务仅支持较少图片，不应直接降低正式抽帧预算绕开，应先核对服务多图配置。合成图像很小，通过不代表真实高分辨率上下文容量足够。

准备小模型环境、GPU 和视频后：

```bash
./run.sh doctor
./run.sh preflight --cases /data/cases.json --output runs/preflight-01
./run.sh run --cases /data/cases.json --output runs/qwen397b-smoke-01
```

建议先用 3–5 道真实题检查完整闭环，再增加队列。`preflight` 解码真实视频、构造首个 Planner 输入后停止，不调用远端或加载小模型。`doctor` 检查本地环境，不检查真实服务连通性。输出目录必须不存在。

后台运行：

```bash
nohup ./run.sh run --cases /data/cases.json --output runs/qwen397b-batch01 > batch01.log 2>&1 &
```

同一队列的小模型常驻，跨题复用权重，题目状态独立；队列退出后只清理自己创建的进程。进程启动失败会明确报错。也支持已有、配置完全匹配的 `COSEEK_QWEN_SOCKET`，不会关闭外部拥有的服务。

## 3. 题单格式

```json
[
  {
    "id": "lvb_001",
    "video": "/data/videos/example.mp4",
    "subtitles": "/data/subtitles/example.srt",
    "question": "原题，不含答案标签",
    "choices": ["选项一", "选项二", "选项三", "选项四", "选项五"],
    "answer": "B"
  }
]
```

支持 2–5 个选项。MLVU 无字幕时省略 `subtitles`；LVB 原始 JSON 字幕需要先按时间戳转为 SRT。`answer` 可省略，仅用于评分，不进入模型输入。ID 使用字母、数字、下划线、点或短横线。

沿用当前 P161 的字幕窗口处理，没有在此次部署中新增字幕压缩。长字幕可能导致输入成本高或超上下文，需要先检查真实请求。

## 4. 服务侧兼容性

- 必须保留视觉编码器，不能以 `--language-model-only` 启动。
- 必须允许 CoSeek 所需的多张图片；可先用 `--limit-mm-per-prompt '{"image":32}'` 对应本包 smoke probe，正式输入仍需按实际日志核对上限。
- 模型的原生上下文长度不等于当前服务的 `--max-model-len`。需要同时容纳字幕、状态、图像和 completion 上限；本包不擅自修改该参数。
- Qwen 思考开关由 `VLLM_ENABLE_THINKING=server|true|false` 控制。默认 `server` 不覆盖启动设置；不会把 GPT 的 medium/low 生硬映射为 Qwen 参数。
- 如果开启思考，应让服务正确分离 reasoning 与 final content；vLLM 官方 Qwen3.5 示例使用 `--reasoning-parser qwen3`。本包遇到空 final content、截断或未分离的 `<think>` 会报错，不把推理文本当作核验回执。
- 默认继续请求 JSON object。若特定服务版本不支持，可显式设 `VLLM_JSON_MODE=false` 做兼容性诊断；这会改变输出约束，应记录后单独验证，不能当作与原设置相同。
- 主要 Planner 路径使用 JSON action。若启用原生 tool calling 分支，还需要服务对应的 tool parser 配置；不能凭纯文本 probe 认定已支持全部工具分支。

参考：[vLLM 官方 Qwen3.5 部署说明](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html)、[PyTorch 官方 CUDA wheel 版本](https://pytorch.org/get-started/previous-versions/)。现有服务的具体参数仍以用户的启动命令为准。

## 5. 保持与改变

保持原题、Prompt、Overview 架构、图片构造、max_steps=20、采样 temperature=1/seed=42、32,768 completion 上限、100k/150k Token 调度阈值和 Capsule 预算。不会把 397B 额外变成搜索模型。

部署适配集中在 `scripts/portable.py`、`transport.py` 和 `vllm_adapter.py`：移除 GPT 固定模型名检查，统一强模型路由，使用 vLLM 的 `max_tokens`，不发送 GPT `reasoning_effort`。实际线上的请求体和服务 usage 均记录。

同一 Token 阈值在 GPT 与 Qwen 下并不意味着相同的信息容量或推理量；原 Capsule 的文本估计仍沿用现有 tokenizer，服务 usage 才是远端实际计数。不要把模型替换理解为同精度替换。

失败策略：单请求 400 秒，SDK 重试 0，全题最多两次额外重试。不可重试失败／耗尽后跳题，队列继续；不按答案正确与否重跑。格式错误也记录为失败，不默默回退到 GPT。每题原有总时间限制与日志保留。

## 6. 当前验证边界

已完成：源码哈希检查、7 项配置／SDK wire／usage 测试、真实视频＋五选项＋SRT 的离线首请求预检（0 API 请求、未加载小模型）、Shell 语法与密钥模式扫描。

未完成：公司内网服务真实调用、公司环境从零安装、B200 上小模型实际加载、397B 全流程精度／时延比较。本机访问目标端点连接被重置，原实验服务器访问超时，不能据此判定 vLLM 服务不可用。请在有公司内网连通性的机器先执行 `probe`。

包不含视频、模型权重、真实密钥、历史回包或已安装环境。`provenance/` 记录源码来源与哈希；`VERIFICATION.md` 记录检查结果。
