# Grid-LLM

> 电网垂直领域大模型微调、GRPO 对齐与高性能部署 — 学习版

## 项目目标

跑通完整 LLM 后训练 + 部署链路：

```
原始数据 → QLoRA SFT → GRPO 对齐 → AWQ 量化 → vLLM 部署
```

用中文数学数据替代电网数据，单张 RTX 5090（32GB）完成全流程。

## 快速开始

### AutoDL 环境（推荐）

```bash
# 1. 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.cargo/env

# 2. 创建环境（uv 自动隔离，不污染系统 Python）
cd grid-llm
uv sync

# 3. 一键跑全流程（自动检测数据盘 /root/autodl-tmp/）
bash run_all.sh
```

> **AutoDL 要点**：所有大文件（模型、数据集、产出物）自动存到 `/root/autodl-tmp/grid-llm/`，系统盘只放代码。

### 本地环境

```bash
# 改 config.yaml 第一行: base_dir: "./"
sed -i 's|/root/autodl-tmp/grid-llm|./|' config/config.yaml
uv sync
bash run_all.sh
```

### 分阶段跑

```bash
uv run python data/download.py           # 下载 CMATH + ape210k
uv run python data/preprocess.py         # 预处理 + 统计数据
uv run python stage1_sft/train.py        # QLoRA SFT (~2h)
uv run python stage2_grpo/train.py       # GRPO 对齐 (~4h)
uv run python stage3_awq/quantize.py     # AWQ INT4 (~30min)
bash stage4_vllm/serve.sh               # vLLM 部署 + 压测
```

## 项目结构

```
grid-llm/
├── README.md                 # 你在这
├── CLAUDE.md                 # AI 助手指令
├── pyproject.toml            # uv 项目配置 & 依赖
├── run_all.sh                # 一键全流程
├── config/config.yaml        # 全局训练配置
├── data/                     # 数据下载 & 预处理
├── stage1_sft/               # QLoRA 微调
├── stage2_grpo/              # GRPO 对齐
├── stage3_awq/               # AWQ 量化
├── stage4_vllm/              # vLLM 部署 & 压测
├── eval/                     # 统一评估
├── utils/                    # 公共工具
└── docs/iterations/          # 迭代文档（spec/plan/task/checklist）
```

## 技术栈

| 阶段 | 框架 | 作用 |
|------|------|------|
| 环境 | uv | 隔离虚拟环境，不污染系统 Python |
| SFT | LLaMA-Factory / peft | QLoRA 4-bit 微调 |
| GRPO | TRL GRPOTrainer + vLLM | 组内相对优势策略优化 |
| AWQ | autoawq | INT4 激活感知量化 |
| vLLM | vllm | PagedAttention 高性能推理 |

## 模型 & 数据

- **基座模型**: Qwen2.5-7B-Instruct
- **GPU**: RTX 5090（32GB VRAM）
- **数据集**: ape210k + CMATH（中文数学，≥5000 条）
- **奖励信号**: 数学答案正确性（规则匹配）

## 迭代文档

每次新功能/改动按 mew-spec 流程产出四份文档，放在 `docs/iterations/<编号>-<名称>/`：

| 编号 | 内容 | 状态 |
|------|------|------|
| 001 | 数学微调全流程（SFT + GRPO + AWQ + vLLM）| 🚧 进行中 |
