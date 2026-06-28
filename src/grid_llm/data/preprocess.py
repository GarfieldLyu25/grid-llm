"""数据预处理：清洗、统一格式化、train/eval/grpo_eval 划分。

输入: data/raw/raw_combined.json（download.py 产出）
输出: data/train.jsonl, data/eval.jsonl, data/grpo_eval.jsonl

每条数据格式: {"instruction": "题目", "output": "逐步推理：\\n...\\n\\n最终答案：xxx"}
"""

import json
import random
import re
from pathlib import Path

from grid_llm.utils.common import (
    load_config, setup_logging, timer,
    MIN_INSTRUCTION_LEN, MAX_INSTRUCTION_LEN, MIN_OUTPUT_LEN,
)

logger = setup_logging("preprocess")

# 推理模板 —— 原始数据没有推理步骤时，嵌入一个最简推理提示
REASONING_TEMPLATES = [
    "逐步推理：\n1. 仔细读题，明确已知条件和问题\n2. 列式计算\n3. 得出结果\n\n最终答案：{answer}",
    "解题过程：\n1. 分析题目给出的条件\n2. 选择合适的方法计算\n3. 验证结果\n\n最终答案：{answer}",
    "推理：\n根据题意，\n所以\n\n最终答案：{answer}",
]


def normalize_sample(sample: dict) -> dict | None:
    """把不同来源的样本统一成 {instruction, output}。

    如果 output 里已有推理步骤就保留；如果只有答案就套模板。
    """
    question = sample.get("question", "").strip()
    answer = sample.get("answer", "").strip()

    if not question or not answer:
        return None

    # 原始 answer 已有推理步骤 → 保留
    has_reasoning = any(
        kw in answer
        for kw in ["步骤", "推理", "解法", "分析", "过程", "思路", "\n", "因为", "所以"]
    )

    if has_reasoning:
        if "最终答案" not in answer and "答案" not in answer[-20:]:
            nums = re.findall(r"-?\d+\.?\d*", answer)
            if nums:
                answer = answer.rstrip() + f"\n\n最终答案：{nums[-1]}"
        output = answer
    else:
        # 只有答案，套推理模板
        template = random.choice(REASONING_TEMPLATES)
        output = template.format(answer=answer)

    return {"instruction": question, "output": output}


def deduplicate(samples: list[dict]) -> list[dict]:
    """按 instruction 去重，保留第一条。"""
    seen: set[str] = set()
    deduped: list[dict] = []
    dup_count = 0
    for s in samples:
        key = s["instruction"].strip()
        if key not in seen:
            seen.add(key)
            deduped.append(s)
        else:
            dup_count += 1
    logger.info(f"去重：移除 {dup_count} 条重复")
    return deduped


def main():
    config = load_config()
    raw_dir = Path(config["data"]["raw_dir"])
    raw_file = raw_dir / "raw_combined.json"

    if not raw_file.exists():
        logger.error(f"原始数据不存在: {raw_file}，请先运行 data/download.py")
        return

    with open(raw_file, "r", encoding="utf-8") as f:
        raw_samples = json.load(f)
    logger.info(f"加载原始数据: {len(raw_samples)} 条")

    with timer("数据处理", logger):
        samples = []
        bad_count = 0
        for s in raw_samples:
            norm = normalize_sample(s)
            if norm:
                samples.append(norm)
            else:
                bad_count += 1
        logger.info(f"有效: {len(samples)} 条，丢弃: {bad_count} 条")

        samples = deduplicate(samples)

        # 过滤太短 / 太长
        samples = [
            s for s in samples
            if MIN_INSTRUCTION_LEN <= len(s["instruction"]) <= MAX_INSTRUCTION_LEN
            and len(s["output"]) >= MIN_OUTPUT_LEN
        ]
        logger.info(f"长度过滤后: {len(samples)} 条")

        # 打乱（局部 Random 实例，不污染全局状态）
        rng = random.Random(42)
        rng.shuffle(samples)

        min_samples = config["data"].get("min_samples", 5000)
        if len(samples) < min_samples:
            logger.warning(f"样本量 {len(samples)} < {min_samples}，建议检查数据源")

        eval_ratio = config["data"]["eval_ratio"]
        grpo_eval_size = config["data"]["grpo_eval_size"]
        total = len(samples)

        eval_count = int(total * eval_ratio)
        train_count = total - eval_count

        train_samples = samples[:train_count]
        eval_samples = samples[train_count:]
        grpo_eval_samples = eval_samples[:grpo_eval_size]

    # 保存
    def save_jsonl(path: str, data: list[dict]):
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with timer("保存文件", logger):
        save_jsonl(config["data"]["train_file"], train_samples)
        save_jsonl(config["data"]["eval_file"], eval_samples)
        save_jsonl(config["data"]["grpo_eval_file"], grpo_eval_samples)

    avg_inst = sum(len(s["instruction"]) for s in samples) / len(samples)
    avg_out = sum(len(s["output"]) for s in samples) / len(samples)

    logger.info("=" * 50)
    logger.info("数据集预处理完成")
    logger.info(f"  原始: {len(raw_samples):,} → 有效: {len(samples):,}")
    logger.info(f"  train={len(train_samples):,}  eval={len(eval_samples):,}  "
                f"grpo_eval={len(grpo_eval_samples):,}")
    logger.info(f"  avg instruction: {avg_inst:.0f} chars  avg output: {avg_out:.0f} chars")


if __name__ == "__main__":
    main()
