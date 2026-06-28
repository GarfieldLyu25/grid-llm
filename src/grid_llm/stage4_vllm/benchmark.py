"""vLLM 吞吐压测。

对比 vLLM AWQ 模型 vs 原生 Transformers 推理。

用法:
  bash src/grid_llm/stage4_vllm/serve.sh    # 先启动 vLLM
  uv run python src/grid_llm/stage4_vllm/benchmark.py
"""

import json
import time
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import torch
from tqdm import tqdm

from grid_llm.utils.common import (
    load_config, setup_logging, timer,
    SYSTEM_PROMPT, apply_chat_template,
    DEFAULT_MAX_NEW_TOKENS, DEFAULT_TEMPERATURE,
)

logger = setup_logging("benchmark")


def load_prompts(file_path: str, num: int = 100) -> list[str]:
    """加载数学问题作为测试 prompts。"""
    prompts: list[str] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if len(prompts) >= num:
                break
            if line.strip():
                d = json.loads(line)
                prompts.append(d["instruction"])
    return prompts


def benchmark_vllm(url: str, prompts: list[str], concurrency: int = 1) -> dict:
    """压测 vLLM API。"""
    logger.info(f"vLLM 压测: {len(prompts)} prompts, 并发={concurrency}")

    results: list[dict] = []
    latencies: list[float] = []

    def send_request(prompt: str) -> tuple[float, int, int]:
        start = time.perf_counter()
        resp = requests.post(
            f"{url}/v1/completions",
            json={
                "model": "default",
                "prompt": prompt,
                "max_tokens": DEFAULT_MAX_NEW_TOKENS,
                "temperature": DEFAULT_TEMPERATURE,
            },
            timeout=120,
        )
        resp.raise_for_status()
        elapsed = time.perf_counter() - start
        data = resp.json()
        tokens = data["usage"]["completion_tokens"]
        prompt_tokens = data["usage"]["prompt_tokens"]
        return elapsed, tokens, prompt_tokens

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(send_request, p) for p in prompts]
        for future in tqdm(as_completed(futures), total=len(futures), desc="vLLM 压测"):
            try:
                elapsed, tokens, prompt_tokens = future.result()
                latencies.append(elapsed)
                results.append({"latency": elapsed, "tokens": tokens, "prompt_tokens": prompt_tokens})
            except Exception as e:
                logger.error(f"请求失败: {e}")

    total_tokens = sum(r["tokens"] for r in results)
    total_time = sum(latencies)
    throughput = total_tokens / total_time if total_time > 0 else 0

    return {
        "total_requests": len(results),
        "total_tokens": total_tokens,
        "total_time_s": total_time,
        "throughput_tokens_per_sec": throughput,
        "avg_latency_s": sum(latencies) / len(latencies) if latencies else 0,
        "p50_latency_s": sorted(latencies)[len(latencies) // 2] if latencies else 0,
        "p95_latency_s": sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0,
    }


def benchmark_transformers(model_path: str, prompts: list[str]) -> dict:
    """压测原生 Transformers 推理（基线）。"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Transformers 压测（基线）...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    total_tokens = 0
    latencies: list[float] = []

    for prompt in tqdm(prompts, desc="Transformers"):
        text = apply_chat_template(tokenizer, prompt, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
                temperature=DEFAULT_TEMPERATURE,
                do_sample=False,
            )
        elapsed = time.perf_counter() - start

        tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
        total_tokens += tokens
        latencies.append(elapsed)

    total_time = sum(latencies)
    throughput = total_tokens / total_time if total_time > 0 else 0

    return {
        "total_requests": len(prompts),
        "total_tokens": total_tokens,
        "total_time_s": total_time,
        "throughput_tokens_per_sec": throughput,
        "avg_latency_s": sum(latencies) / len(latencies),
        "p50_latency_s": sorted(latencies)[len(latencies) // 2],
    }


def main():
    parser = argparse.ArgumentParser(description="vLLM 吞吐压测")
    parser.add_argument("--url", type=str, default="http://localhost:8000")
    parser.add_argument("--num", type=int, default=100, help="压测 prompt 数量")
    parser.add_argument("--concurrency", type=int, default=4, help="并发数")
    parser.add_argument("--skip-transformers", action="store_true", help="跳过基线")
    args = parser.parse_args()

    config = load_config()

    # 加载 prompts
    eval_file = config["data"]["eval_file"]
    prompts = load_prompts(eval_file, num=args.num)
    logger.info(f"加载 {len(prompts)} 条 prompts")

    # vLLM 压测
    logger.info("=" * 40)
    logger.info("压测 vLLM AWQ")
    vllm_result = benchmark_vllm(args.url, prompts, concurrency=args.concurrency)

    # Transformers 基线
    transformers_result = None
    speedup = 0.0
    if not args.skip_transformers:
        logger.info("=" * 40)
        logger.info("压测 Transformers 基线 (GRPO FP16)")
        model_path = config["grpo"]["model_dir"]
        if not Path(model_path).exists():
            logger.warning(f"GRPO 模型不存在: {model_path}，回退 SFT")
            model_path = config["sft"]["merged_dir"]
        transformers_result = benchmark_transformers(
            model_path, prompts[:min(20, len(prompts))]
        )

    # 报告
    logger.info("=" * 60)
    logger.info("吞吐压测报告")
    logger.info(f"Prompts: {args.num}, 并发: {args.concurrency}")
    logger.info(f"vLLM AWQ: {vllm_result['throughput_tokens_per_sec']:.1f} tok/s "
                f"(延迟 {vllm_result['avg_latency_s']:.2f}s, "
                f"P50={vllm_result['p50_latency_s']:.2f}s, "
                f"P95={vllm_result['p95_latency_s']:.2f}s)")

    if transformers_result:
        logger.info(f"Transformers FP16: {transformers_result['throughput_tokens_per_sec']:.1f} tok/s "
                    f"(延迟 {transformers_result['avg_latency_s']:.2f}s)")
        speedup = vllm_result['throughput_tokens_per_sec'] / transformers_result['throughput_tokens_per_sec']
        logger.info(f"vLLM 加速比: {speedup:.1f}x {'[PASS] >= 2x' if speedup >= 2.0 else '[WARN] < 2x'}")

    # 保存结果到数据盘
    output_dir = Path(config["base_dir"]) / "benchmarks"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "benchmark_result.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "vllm": vllm_result,
            "transformers": transformers_result,
            "speedup": speedup if transformers_result else None,
        }, f, ensure_ascii=False, indent=2)
    logger.info(f"结果已保存: {output_file}")


if __name__ == "__main__":
    main()
