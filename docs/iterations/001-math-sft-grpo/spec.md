# Grid-LLM 数学微调全流程 Spec

## 背景

简历项目 Grid-LLM 的完整链路是：电网数据集 → QLoRA SFT → GRPO 对齐 → AWQ 量化 → vLLM 部署。目前没有真实电网数据，目标是用开源数据替代，完整跑通全流程学习技术。

## 目标

1. **跑通全链路**：从零数据到量化部署，每一个中间产物都可检查、可复现
2. **理解每步为什么**：不只要代码能跑，更要理解 SFT 解决什么问题、GRPO 为什么比 PPO 省显存、AWQ 怎么做到 4-bit 不掉精度、vLLM 吞吐为什么高
3. **5090 一次搞定**：所有阶段在单张 RTX 5090（32GB）上完成，不需要多卡

## 功能需求

- F1: 数学数据集构建 — 下载开源中文数学数据集（ape210k / CMATH），统一格式化，划分 train/eval/grpo_eval
- F2: QLoRA SFT 微调 — 基于 Qwen2.5-7B-Instruct，4-bit NF4 + LoRA 增量训练
- F3: GRPO 数学推理增强 — 使用 DeepSeek API 作为裁判型奖励模型，按答案正确性、推理质量、格式规范给 0~1 分；训练仍采用 LoRA 增量更新，组内相对优势估计，移除 Critic
- F4: AWQ INT4 量化 — 量化前后精度对比，验证损失可接受
- F5: vLLM 高性能部署 — PagedAttention + 连续批处理，吞吐压测对比
- F6: 全流程可复现 — 每阶段产出独立保存，`run_all.sh` 一键执行

## 非功能需求

- N1: 显存约束 — 任意时刻峰值 ≤ 28GB
- N2: 训练时间可控 — 单阶段最长 ≤ 6 小时
- N3: 可中断恢复 — 每阶段支持 checkpoint 续训
- N4: 产出物独立 — 每阶段产出独立保存
- N5: 可观测 — loss/tensorboard/reward/精度对比均有可视化

## 不做的事

- 不搜集真实电网数据
- 不追求 SOTA 数学准确率
- 不做多轮对话数学推理
- 不做 DPO 对比实验
- 不部署生产级 API 服务

## 验收标准

| 编号 | 验收项 | 验证方式 |
|------|--------|---------|
| AC1 | 数据集 ≥ 5000 条，train/eval 比例 9:1 | `uv run python src/grid_llm/data/preprocess.py && uv run python src/grid_llm/data/stats.py` |
| AC2 | SFT 后 eval loss 收敛，模型按推理格式输出 | tensorboard |
| AC3 | GRPO LoRA 后 math eval ≥ SFT 基线，API 裁判 reward 上升 | `uv run python src/grid_llm/eval/math_eval.py` |
| AC4 | AWQ INT4 后准确率下降 ≤ 3% | `uv run python src/grid_llm/stage3_awq/eval_compare.py` |
| AC5 | vLLM 吞吐 ≥ 2x 原生 Transformers | `uv run python src/grid_llm/stage4_vllm/benchmark.py` |
| AC6 | `run_all.sh` 一键执行，中断后 checkpoint 续跑 | 手动测试 |
