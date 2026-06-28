"""量化前后精度对比。

分别加载 GRPO 原始模型和 AWQ 量化模型，
在 eval 集上跑 math_eval，对比准确率。

用法:
  uv run python src/grid_llm/stage3_awq/eval_compare.py
"""

import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

from grid_llm.utils.common import load_config, setup_logging, timer, extract_answer, answers_equal, SYSTEM_PROMPT

logger = setup_logging("awq-eval")


def load_eval_data(file_path: str) -> list[dict]:
    samples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def evaluate_model(model, tokenizer, eval_data: list[dict], max_samples: int = 200) -> dict:
    """评估模型在 eval 集上的数学准确率。"""
    correct = 0
    total = 0
    failures = []

    for sample in tqdm(eval_data[:max_samples], desc="评估中"):
        prompt = sample["instruction"]
        ground_truth = extract_answer(sample["output"])
        if ground_truth is None:
            continue

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.0,  # greedy 评估
                do_sample=False,
            )

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        pred = extract_answer(response)

        if pred is not None and answers_equal(pred, ground_truth):
            correct += 1
        else:
            failures.append({
                "prompt": prompt[:80],
                "expected": ground_truth,
                "got": pred,
                "response": response[:200],
            })

        total += 1

    accuracy = correct / total if total > 0 else 0
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "failures": failures[:5],  # 只保留前 5 条失败案例
    }


def main():
    config = load_config()

    eval_data = load_eval_data(config["data"]["eval_file"])
    logger.info(f"eval 数据: {len(eval_data)} 条")

    # 评估原始 GRPO 模型
    logger.info("--- 评估 GRPO 原始模型 ---")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    grpo_path = config["grpo"].get("merged_dir") or config["grpo"]["model_dir"]
    grpo_model = AutoModelForCausalLM.from_pretrained(
        grpo_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    grpo_tokenizer = AutoTokenizer.from_pretrained(grpo_path)

    with timer("GRPO 模型评估", logger):
        grpo_result = evaluate_model(grpo_model, grpo_tokenizer, eval_data)

    del grpo_model
    torch.cuda.empty_cache()

    # 评估 AWQ 量化模型
    logger.info("--- 评估 AWQ 量化模型 ---")
    from awq import AutoAWQForCausalLM

    awq_path = config["awq"]["quantized_dir"]
    awq_model = AutoAWQForCausalLM.from_quantized(awq_path, fuse_layers=True)
    awq_tokenizer = AutoTokenizer.from_pretrained(awq_path)

    with timer("AWQ 模型评估", logger):
        awq_result = evaluate_model(awq_model, awq_tokenizer, eval_data)

    # 输出对比报告
    print(f"\n{'='*60}")
    print(f"  量化精度对比")
    print(f"{'='*60}")
    print(f"  GRPO 原始:  {grpo_result['accuracy']:.2%}  ({grpo_result['correct']}/{grpo_result['total']})")
    print(f"  AWQ INT4:   {awq_result['accuracy']:.2%}  ({awq_result['correct']}/{awq_result['total']})")
    drop = grpo_result['accuracy'] - awq_result['accuracy']
    print(f"  精度损失:    {drop:.2%}")
    print(f"{'='*60}")

    if drop > 0.03:
        print(f"\n  ⚠️ 精度损失 {drop:.2%} 超过 3% 阈值")
    else:
        print(f"\n  ✅ 精度损失在可接受范围内")

    # 展示失败案例
    if awq_result['failures']:
        print(f"\n--- AWQ 失败案例 (前 5 条) ---")
        for i, f in enumerate(awq_result['failures']):
            print(f"\n  [{i+1}] {f['prompt']}")
            print(f"  期望: {f['expected']} → 实际: {f['got']}")


if __name__ == "__main__":
    main()
