"""数据集统计：条数、长度分布、示例。

用法:
  uv run python src/grid_llm/data/stats.py
  uv run python src/grid_llm/data/stats.py --examples
"""

import json
import sys
from pathlib import Path

from grid_llm.utils.common import load_config, setup_logging, extract_answer

logger = setup_logging("stats")


def main():
    config = load_config()
    files = {
        "train": config["data"]["train_file"],
        "eval": config["data"]["eval_file"],
        "grpo_eval": config["data"]["grpo_eval_file"],
    }

    logger.info("=" * 50)
    logger.info("数据集统计")
    logger.info("=" * 50)

    total = 0
    for name, path in files.items():
        p = Path(path)
        if not p.exists():
            logger.warning(f"{name}: 文件不存在 ({path})")
            continue

        with open(path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f if line.strip()]

        total += len(data)
        inst_lens = [len(d["instruction"]) for d in data]
        out_lens = [len(d["output"]) for d in data]

        logger.info(f"[{name}] {len(data):>8,} 条")
        logger.info(f"  instruction: min={min(inst_lens)}, max={max(inst_lens)}, "
                    f"avg={sum(inst_lens) / len(inst_lens):.0f}")
        logger.info(f"  output:      min={min(out_lens)}, max={max(out_lens)}, "
                    f"avg={sum(out_lens) / len(out_lens):.0f}")

        answers = [extract_answer(d["output"]) for d in data]
        answers = [a for a in answers if a is not None]
        logger.info(f"  可提取答案: {len(answers)}/{len(data)} "
                    f"({100 * len(answers) / len(data):.1f}%)")

    logger.info(f"总计: {total:,} 条")

    if "--examples" in sys.argv:
        logger.info("--- 示例 ---")
        with open(config["data"]["train_file"], "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 3:
                    break
                d = json.loads(line)
                logger.info(f"[{i + 1}] {d['instruction'][:100]}...")
                logger.info(f"    → {d['output'][:150]}...")


if __name__ == "__main__":
    main()
