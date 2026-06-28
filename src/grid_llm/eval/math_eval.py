"""统一数学评估脚本。

对指定模型在 eval 集上跑全量推理，计算准确率。
eval 数据路径默认从 config.yaml 读取。

用法:
  uv run python src/grid_llm/eval/math_eval.py --model <path> --max 200
  uv run python src/grid_llm/eval/math_eval.py --model <path> --awq --max 200
"""

import json
import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from grid_llm.utils.common import (
    load_config, setup_logging, timer, extract_answer, answers_equal,
    SYSTEM_PROMPT, apply_chat_template,
    DEFAULT_MAX_NEW_TOKENS, DEFAULT_TEMPERATURE,
)

logger = setup_logging("math-eval")


def load_model(model_path: str, use_awq: bool = False):
    """加载模型，自动处理 AWQ 格式。"""
    if use_awq:
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer
        model = AutoAWQForCausalLM.from_quantized(model_path, fuse_layers=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer


def validate_path_within(path_str: str, allowed_dir: str) -> Path:
    """校验路径在允许目录内，防止路径穿越。"""
    p = Path(path_str).resolve()
    allowed = Path(allowed_dir).resolve()
    if str(p) != str(allowed) and not str(p).startswith(str(allowed) + "/"):
        raise ValueError(f"路径不在允许范围内: {p}")
    return p


def main():
    config = load_config()

    parser = argparse.ArgumentParser(description="数学评估")
    parser.add_argument("--model", type=str, required=True, help="模型路径")
    parser.add_argument("--eval-file", type=str, default=config["data"]["eval_file"],
                        help="eval 数据路径")
    parser.add_argument("--max", type=int, default=200, dest="max_samples")
    parser.add_argument("--awq", action="store_true", help="模型是 AWQ 量化格式")
    parser.add_argument("--output", type=str, default=None, help="结果保存路径")
    args = parser.parse_args()

    # 路径安全：--output 必须在 base_dir 内
    base_dir = config["base_dir"]
    output_path = None
    if args.output:
        output_path = validate_path_within(args.output, base_dir)

    # 加载 eval 数据
    eval_data = []
    with open(args.eval_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                eval_data.append(json.loads(line))

    logger.info(f"加载 {len(eval_data)} 条 eval 数据，评估前 {args.max_samples} 条")

    # 加载模型
    model, tokenizer = load_model(args.model, use_awq=args.awq)

    # 评估
    correct = 0
    total = 0
    results = []

    with timer("评估", logger):
        for sample in tqdm(eval_data[:args.max_samples], desc="推理"):
            prompt = sample["instruction"]
            gt = extract_answer(sample["output"])
            if gt is None:
                continue

            text = apply_chat_template(tokenizer, prompt, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
                    temperature=DEFAULT_TEMPERATURE,
                    do_sample=False,
                )

            response = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
            pred = extract_answer(response)
            is_correct = pred is not None and answers_equal(pred, gt)

            if is_correct:
                correct += 1
            total += 1

            results.append({
                "prompt": prompt[:100],
                "expected": gt,
                "predicted": pred,
                "correct": is_correct,
            })

    acc = correct / total if total > 0 else 0

    logger.info(f"模型: {args.model}")
    logger.info(f"准确率: {acc:.2%} ({correct}/{total})")

    if output_path:
        output = {
            "model": args.model,
            "accuracy": acc,
            "correct": correct,
            "total": total,
            "results": results,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        logger.info(f"结果已保存: {output_path}")


if __name__ == "__main__":
    main()
