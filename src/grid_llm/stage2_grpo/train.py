"""Stage 2: GRPO 数学推理增强。

用 TRL GRPOTrainer，自定义数学正确性奖励函数，
组内相对优势估计更新策略。移除 PPO 的 Critic 模型。

产出: 数据盘 stage2_grpo/output/grpo_model/
"""

import json
from pathlib import Path

import torch
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

from grid_llm.stage2_grpo.reward import build_reward_func
from grid_llm.utils.common import (
    load_config, setup_logging, timer, extract_answer,
    SYSTEM_PROMPT, apply_chat_template,
)

logger = setup_logging("grpo")


def load_grpo_data(file_path: str) -> tuple[Dataset, dict[str, str]]:
    """加载 GRPO 数据，构建 prompt → ground_truth 映射。

    返回 (Dataset, {prompt_text: ground_truth_answer})
    """
    samples = []
    prompt_to_gt: dict[str, str] = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            prompt = d["instruction"]
            samples.append({"prompt": prompt})
            ans = extract_answer(d["output"])
            prompt_to_gt[prompt] = ans if ans else "0"

    logger.info(f"GRPO 数据: {len(samples)} 条 prompt，{len(prompt_to_gt)} 条 ground_truth")
    return Dataset.from_list(samples), prompt_to_gt


def main():
    config = load_config()
    grpo_cfg = config["grpo"]
    output_dir = Path(grpo_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = config["sft"]["merged_dir"]

    # 加载数据 — 返回 prompt→gt 映射
    ds, prompt_to_gt = load_grpo_data(config["data"]["grpo_eval_file"])

    # 用 prompt 文本精确匹配 ground_truth
    reward_func = build_reward_func(prompt_to_gt)

    logger.info(f"加载 SFT 模型: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize
    def tokenize_prompt(example):
        return apply_chat_template(tokenizer, example["prompt"], add_generation_prompt=True)

    train_cfg = grpo_cfg["training"]

    grpo_config = GRPOConfig(
        output_dir=str(output_dir),
        num_generations=train_cfg["num_generations"],
        max_prompt_length=train_cfg["max_prompt_length"],
        max_completion_length=train_cfg["max_completion_length"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        num_train_epochs=train_cfg["num_train_epochs"],
        beta=train_cfg["beta"],
        logging_steps=train_cfg["logging_steps"],
        save_steps=train_cfg["save_steps"],
        bf16=train_cfg["bf16"],
        seed=train_cfg["seed"],
        report_to="tensorboard",
        run_name="grid-llm-grpo",
    )

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=ds,
        reward_funcs=[reward_func],
        processing_class=tokenizer,
    )

    logger.info(f"开始 GRPO 训练 (num_generations={train_cfg['num_generations']}, "
                f"beta={train_cfg['beta']}, lr={train_cfg['learning_rate']})")

    with timer("GRPO 训练", logger):
        trainer.train()

    # 保存
    model_dir = Path(grpo_cfg["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    logger.info(f"GRPO 模型已保存到 {model_dir}")

    mem = torch.cuda.max_memory_allocated() / 1024**3
    logger.info(f"训练完成，峰值显存: {mem:.1f} GB")


if __name__ == "__main__":
    main()
