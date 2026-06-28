#!/bin/bash
# Stage 4: vLLM 部署 — 启动 AWQ 量化模型的 OpenAI 兼容 API 服务
#
# 用法:
#   bash src/grid_llm/stage4_vllm/serve.sh          # 启动服务
#   bash src/grid_llm/stage4_vllm/serve.sh --bench  # 启动后再跑压测
#
# 服务地址: http://localhost:8000
# API 文档: http://localhost:8000/docs

set -euo pipefail

# 从配置读模型路径（兼容 AutoDL 数据盘）
MODEL_PATH=$(uv run python -c "from grid_llm.utils.common import load_config; c=load_config(); print(c['awq']['quantized_dir'])")
PORT=8000
HOST="0.0.0.0"

# 检查模型是否存在
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ 量化模型不存在: $MODEL_PATH"
    echo "   请先运行: uv run python src/grid_llm/stage3_awq/quantize.py"
    exit 1
fi

echo "========================================="
echo "  vLLM 部署 — AWQ INT4"
echo "========================================="
echo "  模型: $MODEL_PATH"
echo "  地址: http://${HOST}:${PORT}"
echo "========================================="

# 启动 vLLM API 服务
# awq_marlin 在 Blackwell (5090) 上性能更好
uv run python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --quantization awq_marlin \
    --dtype float16 \
    --host "$HOST" \
    --port "$PORT" \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.90 \
    --trust-remote-code &

VLLM_PID=$!
echo "  vLLM PID: $VLLM_PID"

# 等待服务就绪
echo "  等待服务就绪 ..."
for i in $(seq 1 60); do
    if curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; then
        echo "  ✅ 服务就绪"
        break
    fi
    sleep 2
done

# 如果传了 --bench，自动跑压测
if [ "${1:-}" = "--bench" ]; then
    echo ""
    echo "  开始压测 ..."
    uv run python src/grid_llm/stage4_vllm/benchmark.py
fi

echo ""
echo "  服务运行中 (PID: $VLLM_PID)"
echo "  停止: kill $VLLM_PID"

# 保持前台
wait $VLLM_PID
