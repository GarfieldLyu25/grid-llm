"""GRPO 模型推理 & SFT 对比。

对 SFT 模型和 GRPO 模型分别跑同一条 prompt，肉眼对比质量。
"""

import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from grid_llm.utils.common import load_config, setup_logging, timer, extract_answer, SYSTEM_PROMPT

logger = setup_logging("grpo-inference")


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 512) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def main():
    config = load_config()
    sft_path = config["sft"]["merged_dir"]
    grpo_path = config["grpo"]["model_dir"]

    prompt = "鸡兔同笼，头有 35 个，脚有 94 只，鸡和兔各有多少只？"
    for i, arg in enumerate(sys.argv):
        if arg == "--prompt" and i + 1 < len(sys.argv):
            prompt = sys.argv[i + 1]
            break

    print(f"\n  题目: {prompt}\n{'='*60}")

    for name, path in [("SFT", sft_path), ("GRPO", grpo_path)]:
        if not Path(path).exists():
            logger.warning(f"{name} 模型路径不存在: {path}")
            continue

        logger.info(f"加载 {name} 模型: {path}")
        model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(path)

        with timer(f"{name} 推理", logger):
            response = generate(model, tokenizer, prompt)

        print(f"\n--- {name} 输出 ---")
        print(response)
        ans = extract_answer(response)
        if ans:
            print(f"  📝 提取答案: {ans}")

        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
