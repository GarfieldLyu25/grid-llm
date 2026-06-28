"""Stage 3: AWQ INT4 量化。

对 GRPO 后的模型做 AWQ 4-bit 激活感知量化。
用 eval 数据做校准，产出量化模型。

注意: autoawq 已被官方 deprecated，推荐 llm-compressor。
本脚本保留 autoawq 路径（简单），并注释 llm-compressor 替代方案。

产出: stage3_awq/output/model-awq-4bit/
"""

import json
import sys
from pathlib import Path

import torch

from grid_llm.utils.common import load_config, setup_logging, timer, SYSTEM_PROMPT

logger = setup_logging("awq")


def load_calib_data(file_path: str, tokenizer, max_samples: int = 128) -> list[str]:
    """从 eval 数据加载校准样本，转为 chat template 文本。"""
    samples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": d["instruction"]},
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            samples.append(text.strip())
            if len(samples) >= max_samples:
                break
    return samples


def quantize_with_autoawq(config: dict):
    """使用 autoawq 量化（需要 transformers<=4.51.3）。"""
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    awq_cfg = config["awq"]
    model_path = config["grpo"].get("merged_dir") or config["grpo"]["model_dir"]
    quant_path = awq_cfg["quantized_dir"]

    logger.info(f"加载 GRPO 模型: {model_path}")
    model = AutoAWQForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        use_cache=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # 加载校准数据
    calib_data = load_calib_data(
        config["data"]["eval_file"],
        tokenizer,
        max_samples=awq_cfg.get("calib_size", 128),
    )
    logger.info(f"校准数据: {len(calib_data)} 条")

    quant_config = {
        "zero_point": awq_cfg.get("zero_point", True),
        "q_group_size": awq_cfg.get("group_size", 128),
        "w_bit": awq_cfg.get("bits", 4),
        "version": "GEMM",
    }

    logger.info("开始 AWQ 量化 ...")
    with timer("AWQ 量化", logger):
        model.quantize(
            tokenizer,
            quant_config=quant_config,
            calib_data=calib_data,
        )

    # 保存
    Path(quant_path).mkdir(parents=True, exist_ok=True)
    model.save_quantized(quant_path)
    tokenizer.save_pretrained(quant_path)
    logger.info(f"量化模型已保存到 {quant_path}")

    # 打印大小对比
    import os
    orig_size = sum(
        os.path.getsize(os.path.join(model_path, f))
        for f in os.listdir(model_path)
        if f.endswith(".safetensors")
    )
    quant_size = sum(
        os.path.getsize(os.path.join(quant_path, f))
        for f in os.listdir(quant_path)
        if f.endswith(".safetensors")
    )
    logger.info(f"原始大小: {orig_size / 1e9:.1f} GB → 量化后: {quant_size / 1e9:.1f} GB "
                f"({100 * quant_size / orig_size:.0f}%)")


def quantize_with_llmcompressor(config: dict):
    """使用 llm-compressor 量化（推荐，新项目首选）。

    安装: pip install llmcompressor
    """
    logger.warning("llm-compressor 路径暂未实现，请使用 autoawq 路径")
    logger.info("参考: https://github.com/vllm-project/llm-compressor")
    sys.exit(1)


def main():
    config = load_config()
    backend = "autoawq"  # 默认；改成 llmcompressor 切换到新方案

    if backend == "autoawq":
        try:
            quantize_with_autoawq(config)
        except ImportError as e:
            logger.error(f"autoawq 导入失败: {e}")
            logger.info("请安装: pip install autoawq autoawq-kernels transformers==4.51.3")
            logger.info("或使用 llm-compressor: pip install llmcompressor")
            sys.exit(1)
    elif backend == "llmcompressor":
        quantize_with_llmcompressor(config)
    else:
        logger.error(f"未知量化后端: {backend}")


if __name__ == "__main__":
    main()
