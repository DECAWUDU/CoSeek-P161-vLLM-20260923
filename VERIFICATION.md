# 部署验证记录

2026-09-23。基于近期 P161；P162 未合入。所有检查都在隔离目录执行，未修改现有运行代码／Python 环境／vLLM 服务。

- 7/7 单元及 SDK 集成测试通过。SDK 使用临时本机模拟 HTTP 服务，验证 `/v1/models`、实际 JSON 请求体、thinking 参数、模型路由及 usage 写入，不表示公司 vLLM 已通过。
- 真实视频离线预检通过：五选项＋SRT，`preflight_passed`，API 请求数 0，无本地模型加载。
- `runtime/` 与来源冻结目录文件 SHA256 对应；核心算法文件未修改。
- `setup.sh`、`run.sh` Shell 语法检查通过。
- 文本源文件未发现真实 `sk-...` 密钥模式；归档不包含 `.env`、回包、视频或模型权重。
- 未测试全新环境依赖安装与 B200 执行。安装脚本使用单独虚拟环境，需从安装网络取得依赖。
- 目标 `http://10.90.79.125:8077/v1/models`：本地连接重置，原实验服务器连接超时。尚未取得模型 ID、max_model_len、鉴权要求或启动参数。
- `probe` 和真实 3–5 题仍应在公司内网完成。P161 与 GPT-5 的历史精度不可直接移用到 Qwen3.5-397B。

复核：`./run.sh verify`，`.venvs/main/bin/python -m unittest discover -s tests -v`。
