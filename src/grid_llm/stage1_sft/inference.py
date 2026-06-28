"""SFT 模型推理验证：加载合并模型，跑示例推理。

用法:
  uv run python src/grid_llm/stage1_sft/inference.py                        # 默认示例
  uv run python src/grid_llm/stage1_sft/inference.py --prompt "你的题目"     # 自定义题目
"""

import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from grid_llm.utils.common import load_config, setup_logging, timer, extract_answer, SYSTEM_PROMPT

logger = setup_logging("inference")


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 512) -> str:
    """用对话格式推理。"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response


def main():
    config = load_config()
    model_path = config["sft"]["merged_dir"]

    logger.info(f"加载模型: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 默认示例
    examples = [
        "鸡兔同笼，头有 35 个，脚有 94 只，鸡和兔各有多少只？",
        "小明有 120 元，买书花了 35 元，买文具花了 28 元，还剩多少钱？",
        "一个长方形长 12 米，宽是长的 3/4，求面积。",
    ]

    # 命令行自定义 prompt
    prompt = None
    for i, arg in enumerate(sys.argv):
        if arg == "--prompt" and i + 1 < len(sys.argv):
            prompt = sys.argv[i + 1]
            break

    if prompt:
        examples = [prompt]

    for i, q in enumerate(examples):
        print(f"\n{'='*60}")
        print(f"  [{i+1}] {q}")
        print(f"{'='*60}")
        with timer("推理", logger):
            response = generate(model, tokenizer, q)
        print(response)
        answer = extract_answer(response)
        if answer:
            print(f"\n  📝 提取答案: {answer}")


if __name__ == "__main__":
    main()
