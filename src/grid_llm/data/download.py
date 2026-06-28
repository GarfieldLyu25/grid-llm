"""下载数学数据集：ape210k + CMATH。

AutoDL 适配：
- HF 缓存指向数据盘 /root/autodl-tmp/.cache/huggingface/
- 如果 HF 已有缓存则秒过
- 数据集保存到数据盘

用法:
  uv run python src/grid_llm/data/download.py
"""

import json
import os
from pathlib import Path

from datasets import load_dataset
from grid_llm.utils.common import load_config, setup_logging, timer

logger = setup_logging("download")


def setup_hf_cache(base_dir: str):
    """将 HuggingFace 缓存指向数据盘（避免撑爆系统盘）。"""
    cache_dir = os.path.join(base_dir, ".cache", "huggingface")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["HF_HOME"] = cache_dir
    os.environ["HUGGINGFACE_HUB_CACHE"] = cache_dir
    logger.info(f"HF 缓存目录: {cache_dir}")


def download_ape210k() -> list[dict]:
    """下载 ape210k 中文数学应用题。"""
    logger.info("正在加载 ape210k ...")
    ds = load_dataset("xuefeng/ape210k", split="train")
    samples = []
    for item in ds:
        samples.append({
            "source": "ape210k",
            "question": item["question"],
            "answer": item["answer"],
        })
    logger.info(f"ape210k: {len(samples)} 条")
    return samples


def download_cmath() -> list[dict]:
    """下载 CMATH 中文数学竞赛题。"""
    logger.info("正在加载 CMATH ...")
    ds = load_dataset("TIGER-Lab/CMATH", split="train")
    samples = []
    for item in ds:
        samples.append({
            "source": "cmath",
            "question": item["question"],
            "answer": item["solution"],  # CMATH 字段名不同
        })
    logger.info(f"CMATH: {len(samples)} 条")
    return samples


def main():
    config = load_config()
    base_dir = config["base_dir"].rstrip("/")

    # 设置 HF 缓存到数据盘
    setup_hf_cache(base_dir)

    raw_dir = Path(config["data"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    with timer("全部数据加载", logger):
        all_samples = []

        for ds_name in config["data"]["datasets"]:
            if "ape210k" in ds_name:
                samples = download_ape210k()
            elif "CMATH" in ds_name:
                samples = download_cmath()
            else:
                logger.warning(f"未知数据集: {ds_name}，跳过")
                continue
            all_samples.extend(samples)

        # 保存
        output_path = raw_dir / "raw_combined.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_samples, f, ensure_ascii=False, indent=2)

        # 打印大小
        size_mb = output_path.stat().st_size / 1024**2
        logger.info(f"已保存到 {output_path} ({size_mb:.1f} MB)，总计 {len(all_samples)} 条")


if __name__ == "__main__":
    main()
