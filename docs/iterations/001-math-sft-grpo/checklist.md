# Grid-LLM 数学微调全流程 Checklist

> 每一项通过运行代码或观察行为来验证，聚焦系统行为。

## 实现完整性
- [ ] T1-T11 全部文件已创建（验证：`find src/grid_llm -name "*.py" -o -name "*.sh" -o -name "*.yaml" | wc -l` ≥ 18）
- [ ] 数据预处理输出 train.jsonl ≥ 4500 条（验证：`uv run python src/grid_llm/data/stats.py`）
- [ ] SFT adapter 权重已保存（验证：`ls stage1_sft/output/lora_adapter/adapter_model.safetensors`）
- [ ] GRPO LoRA 与合并模型已保存（验证：`ls stage2_grpo/output/grpo_lora_adapter/` 和 `ls stage2_grpo/output/grpo_merged/`）
- [ ] AWQ 量化模型已保存（验证：`ls stage3_awq/output/model-awq-4bit/`）

## 集成
- [ ] `run_all.sh` 一键执行全流程（验证：`bash run_all.sh` 所有阶段依次完成）
- [ ] 断点续跑：中断 `run_all.sh` 后重新执行，已完成阶段自动跳过
- [ ] 每阶段产出物可独立加载验证（验证：各阶段 inference.py 或 eval 脚本单独运行成功）

## 质量
- [ ] SFT loss 收敛（验证：tensorboard 看 loss 曲线持续下降）
- [ ] GRPO API 裁判 reward 上升（验证：tensorboard 看 reward 均值上升趋势）
- [ ] GRPO eval 准确率 ≥ SFT eval 准确率（验证：`uv run python src/grid_llm/eval/math_eval.py`）
- [ ] AWQ 量化准确率下降 ≤ 3%（验证：`uv run python src/grid_llm/stage3_awq/eval_compare.py`）
- [ ] vLLM 吞吐 ≥ 2x Transformers（验证：`uv run python src/grid_llm/stage4_vllm/benchmark.py`）
- [ ] 训练全程显存峰值 ≤ 28GB（验证：`nvidia-smi` 监控）

## 端到端场景
- [ ] 场景 1：SFT 模型输出逐步推理 + 正确答案（验证：`uv run python src/grid_llm/stage1_sft/inference.py`）
- [ ] 场景 2：GRPO 模型输出质量 ≥ SFT（验证：对比两次推理）
- [ ] 场景 3：vLLM API 正确返回（验证：`curl http://localhost:8000/v1/completions`）
