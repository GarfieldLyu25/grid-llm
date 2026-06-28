# Grid-LLM 数学微调全流程 Plan

## 架构概览

线性 Pipeline，每阶段产出物是下一阶段的输入：

```
数学原始数据
    │
    ▼
[data/] 数据预处理 ────→ train.jsonl + eval.jsonl + grpo_eval.jsonl
    │
    ▼
[stage1_sft/] QLoRA SFT ────→ lora_adapter/ + sft_merged/
    │
    ▼
[stage2_grpo/] GRPO 对齐 ────→ grpo_model/
    │
    ▼
[stage3_awq/] AWQ 量化 ────→ model-awq-4bit/
    │
    ▼
[stage4_vllm/] vLLM 部署 ────→ 吞吐压测报告
```

## 核心数据结构

### 数据集格式（全流程统一）

```json
{
  "instruction": "小明有 12 个苹果，给了小红 3 个，又买了 5 个，现在有几个？",
  "output": "逐步推理：\n1. 小明最初有 12 个苹果\n2. 给了小红 3 个，剩余 12 - 3 = 9 个\n3. 又买了 5 个，共 9 + 5 = 14 个\n\n最终答案：14"
}
```

### GRPO 奖励函数输出

```python
@dataclass
class RewardResult:
    score: float          # 0.0 ~ 1.0
    is_correct: bool      # 最终答案是否正确
    has_reasoning: bool   # 是否有推理步骤
    extracted_answer: str # 从模型输出中提取的答案
    ground_truth: str     # 标准答案
```

## 模块设计

### [模块 A] data/ — 数据预处理

- **职责**：下载 CMATH/ape210k，清洗去重，统一格式，划分 train/eval
- **对外接口**：`data/train.jsonl`, `data/eval.jsonl`, `data/grpo_eval.jsonl`
- **依赖**：无（起点）

### [模块 B] stage1_sft/ — QLoRA SFT

- **职责**：Load Qwen2.5-7B-Instruct → 4-bit NF4 量化 → LoRA 注入 → 训练 → 合并权重
- **框架**：LLaMA-Factory（优先）或 transformers + peft
- **对外接口**：`stage1_sft/output/lora_adapter/` + `stage1_sft/output/sft_merged/`
- **依赖**：data/train.jsonl

### [模块 C] stage2_grpo/ — GRPO 对齐

- **职责**：用 TRL GRPOTrainer，自定义数学正确性 Reward，组内相对优势更新
- **框架**：trl.GRPOTrainer + vLLM（推理加速）
- **对外接口**：`stage2_grpo/output/grpo_model/`
- **依赖**：stage1_sft/output/sft_merged/ + data/grpo_eval.jsonl

### [模块 D] stage3_awq/ — AWQ 量化

- **职责**：GRPO 模型 → AWQ INT4 量化 → 量化前后精度对比
- **框架**：autoawq
- **对外接口**：`stage3_awq/output/model-awq-4bit/`
- **依赖**：stage2_grpo/output/grpo_model/

### [模块 E] stage4_vllm/ — vLLM 部署

- **职责**：加载 AWQ 模型 → 启动 vLLM → 压测对比吞吐
- **框架**：vllm
- **对外接口**：benchmark 报告（终端输出 + JSON）
- **依赖**：stage3_awq/output/model-awq-4bit/

## 模块交互（数据流）

```
[原始数学数据]
      │
      ▼ data/preprocess.py
[train.jsonl / eval.jsonl / grpo_eval.jsonl]
      │
      ├──────────────────────────┐
      ▼                          ▼
stage1_sft/train.py       stage1_sft/eval_loss.py
      │
      ▼
[lora_adapter / sft_merged]
      │
      ▼ stage2_grpo/train.py (reward.py 提供 reward 函数)
[grpo_model]
      │
      ▼ stage3_awq/quantize.py
[model-awq-4bit]
      │
      ▼ stage4_vllm/serve.sh + benchmark.py
[吞吐压测报告]
```

## 文件组织

```
grid-llm/
├── README.md                    # 全流程教程
├── CLAUDE.md                    # AI 助手指令（强制走 mew-spec 流程）
├── requirements.txt
├── run_all.sh                   # 一键执行全流程
│
├── config/
│   └── config.yaml              # 全局配置
│
├── data/
│   ├── download.py              # 从 HuggingFace 下载 CMATH/ape210k
│   ├── preprocess.py            # 清洗、格式转换、train/eval 划分
│   └── stats.py                 # 数据集统计
│
├── stage1_sft/
│   ├── train.py                 # QLoRA SFT 主训练脚本
│   ├── merge.py                 # LoRA 权重合并
│   └── inference.py             # 单条推理验证
│
├── stage2_grpo/
│   ├── reward.py                # 数学正确性奖励函数
│   ├── train.py                 # GRPO 训练
│   └── inference.py             # 推理 + 准确率对比
│
├── stage3_awq/
│   ├── quantize.py              # AWQ 量化
│   └── eval_compare.py          # 量化前后精度对比
│
├── stage4_vllm/
│   ├── serve.sh                 # 启动 vLLM 服务
│   └── benchmark.py             # 吞吐/延迟压测
│
├── eval/
│   └── math_eval.py             # 统一数学评估
│
└── utils/
    ├── common.py                # 公共工具
    └── extract_answer.py        # 答案提取
```

## 技术决策

| 决策点 | 选择 | 理由 |
|--------|------|------|
| SFT 框架 | LLaMA-Factory | QLoRA 一键配置，中文社区活跃 |
| 备选 SFT | transformers + peft | 降级方案 |
| GRPO 框架 | TRL GRPOTrainer | HuggingFace 官方，API 稳定 |
| GRPO 推理后端 | vLLM (inside TRL) | 采样加速 5-10x |
| 奖励函数 | 规则提取 + 字符串比对 | 数学答案明确，无需 reward model |
| 量化 | AWQ | 激活感知，数学推理精度更好 |
| 部署推理 | vLLM | 社区标准，原生支持 AWQ |
| 数学数据集 | ape210k + CMATH | 中文数学，量大稳定 |
| Batch size | SFT: 4, GRPO: 2 | 5090 32GB 安全值 |
| LoRA rank | 64 | 平衡速度与效果 |
| 精度 | BF16 训练，NF4 量化 base | 5090 硬件加速 |
