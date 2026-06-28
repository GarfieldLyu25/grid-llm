#!/bin/bash
# Grid-LLM 全流程一键执行（AutoDL / Linux）
#
# 用法:
#   bash run_all.sh              # 从头执行
#
# 前置: uv sync

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
log_step() { echo -e "${GREEN}[Step $1]${NC} $2"; }
log_skip() { echo -e "${YELLOW}[Skip]${NC}  $1 已存在，跳过"; }
log_info() { echo -e "${BLUE}[Info]${NC} $1"; }
log_err()  { echo -e "${RED}[Error]${NC} $1"; exit 1; }

SRC="src/grid_llm"

# 解析配置路径的函数
cfg_val() {
    uv run python -c "from grid_llm.utils.common import load_config; c=load_config(); print(c$1.replace('{base_dir}', c['base_dir']))"
}

# ---- 环境检查 ----
echo "========================================="
echo "  Grid-LLM 全流程"
echo "  SFT → GRPO → AWQ → vLLM"
echo "========================================="

DATA_DISK="/root/autodl-tmp"
if [ -d "$DATA_DISK" ]; then
    log_info "AutoDL 数据盘: $DATA_DISK"
    export HF_HOME="${DATA_DISK}/.cache/huggingface"
    mkdir -p "$HF_HOME"
fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1 2>/dev/null || true
echo ""

# ---- Step 1: 数据下载 ----
STEP=1; OUT=$(cfg_val "['data']['raw_dir']")/raw_combined.json
if [ -f "$OUT" ]; then log_skip "数据集"
else
    log_step $STEP "下载数据集 (ape210k + CMATH)"
    uv run python "$SRC/data/download.py" || log_err "数据下载失败"
fi

# ---- Step 2: 预处理 ----
STEP=2; OUT=$(cfg_val "['data']['train_file']")
if [ -f "$OUT" ]; then log_skip "预处理数据"
else
    log_step $STEP "数据预处理"
    uv run python "$SRC/data/preprocess.py" || log_err "预处理失败"
    uv run python "$SRC/data/stats.py"
fi

# ---- Step 3: SFT LoRA ----
STEP=3; OUT=$(cfg_val "['sft']['lora_adapter_dir']")/adapter_model.safetensors
if [ -f "$OUT" ]; then log_skip "SFT LoRA adapter"
else
    log_step $STEP "QLoRA SFT 微调 (~2h)"
    uv run python "$SRC/stage1_sft/train.py" || log_err "SFT 训练失败"
fi

# ---- Step 4: 合并权重 ----
STEP=4; OUT=$(cfg_val "['sft']['merged_dir']")/config.json
if [ -f "$OUT" ]; then log_skip "SFT 合并模型"
else
    log_step $STEP "合并 LoRA 权重"
    uv run python "$SRC/stage1_sft/merge.py" || log_err "合并失败"
fi

# ---- Step 5: 奖励函数测试 ----
STEP=5
log_step $STEP "GRPO 奖励函数测试"
uv run python "$SRC/stage2_grpo/reward.py" --test || log_err "奖励函数测试失败"

# ---- Step 6: GRPO 训练 ----
STEP=6; OUT=$(cfg_val "['grpo']['model_dir']")/config.json
if [ -f "$OUT" ]; then log_skip "GRPO 模型"
else
    log_step $STEP "GRPO 对齐训练 (~4h)"
    uv run python "$SRC/stage2_grpo/train.py" || log_err "GRPO 训练失败"
fi

# ---- Step 7: SFT vs GRPO 评估 ----
STEP=7
log_step $STEP "SFT vs GRPO 准确率对比"
uv run python "$SRC/eval/math_eval.py" --model "$(cfg_val "['sft']['merged_dir']")" --max 100
uv run python "$SRC/eval/math_eval.py" --model "$(cfg_val "['grpo']['model_dir']")" --max 100

# ---- Step 8: AWQ 量化 ----
STEP=8; OUT=$(cfg_val "['awq']['quantized_dir']")/config.json
if [ -f "$OUT" ]; then log_skip "AWQ 量化模型"
else
    log_step $STEP "AWQ INT4 量化 (~30min)"
    uv run python "$SRC/stage3_awq/quantize.py" || log_err "AWQ 量化失败"
fi

# ---- Step 9: 量化精度对比 ----
STEP=9
log_step $STEP "量化前后精度对比"
uv run python "$SRC/stage3_awq/eval_compare.py" || echo "  (非致命，继续)"

# ---- Done ----
echo ""
echo "========================================="
echo "  🎉 全流程完成！"
echo "========================================="
echo "  产出: $(cfg_val "['base_dir']")"
echo ""
echo "  vLLM: bash $SRC/stage4_vllm/serve.sh"
echo "  TensorBoard: tensorboard --logdir $(cfg_val "['base_dir']")"
