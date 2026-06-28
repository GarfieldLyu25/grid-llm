# Grid-LLM 数学微调全流程 Task

## 文件清单

| 操作 | 文件 | 职责 |
|------|------|------|
| 新建 | `config/config.yaml` | 可提交配置模板 |
| 本地生成 | `config/config.local.yaml` | 实际运行配置，填写真实 API key，git 忽略 |
| 新建 | `pyproject.toml` | uv 项目配置 & 依赖 |
| 新建 | `src/grid_llm/data/download.py` | 从 HuggingFace 下载 CMATH/ape210k |
| 新建 | `src/grid_llm/data/preprocess.py` | 清洗、格式转换、数据划分 |
| 新建 | `src/grid_llm/data/stats.py` | 数据集统计 |
| 新建 | `src/grid_llm/stage1_sft/train.py` | QLoRA SFT 训练 |
| 新建 | `src/grid_llm/stage1_sft/merge.py` | LoRA 合并 |
| 新建 | `src/grid_llm/stage1_sft/inference.py` | 单条推理验证 |
| 新建 | `src/grid_llm/stage2_grpo/reward.py` | DeepSeek API 裁判奖励函数，保留规则奖励测试 |
| 新建 | `src/grid_llm/stage2_grpo/train.py` | GRPO LoRA 训练 |
| 新建 | `src/grid_llm/stage2_grpo/inference.py` | GRPO 模型推理 |
| 新建 | `src/grid_llm/stage3_awq/quantize.py` | AWQ INT4 量化 |
| 新建 | `src/grid_llm/stage3_awq/eval_compare.py` | 量化前后精度对比 |
| 新建 | `src/grid_llm/stage4_vllm/serve.sh` | vLLM 启动脚本 |
| 新建 | `src/grid_llm/stage4_vllm/benchmark.py` | 吞吐压测 |
| 新建 | `src/grid_llm/eval/math_eval.py` | 统一数学评估 |
| 新建 | `src/grid_llm/utils/common.py` | 公共工具 |
| 新建 | `src/grid_llm/utils/extract_answer.py` | 答案提取 |
| 新建 | `run_all.sh` | 全流程一键执行 |

## 任务列表

- T1: 环境配置和依赖 ✓
- T2: 数据集下载 ✓
- T3: 数据预处理 ✓
- T4: SFT 训练 (QLoRA) ✓
- T5: SFT 模型合并 & 推理 ✓
- T6: GRPO API 裁判奖励函数 ✓
- T7: GRPO LoRA 训练 ✓
- T8: GRPO 模型评估 ✓
- T9: AWQ 量化 ✓
- T10: vLLM 部署 & 压测 ✓
- T11: 全流程串联 ✓

## 执行顺序

```
T1 → T2 → T3 → T4 → T5
                      ↘
                  T6 → T7 → T8
                              ↘
                              T9 → T10 → T11
```
