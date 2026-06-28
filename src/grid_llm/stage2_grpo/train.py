"""Stage 2: GRPO 数学推理增强。

对 SFT 合并模型做 QLoRA + GRPO。奖励函数用 DeepSeek API 裁判，
评判答案正确性 + 推理质量 + 格式规范，返回 0~1 连续分数。

GRPO 训练阶段保存 LoRA adapter，然后合并到 SFT 模型形成完整 GRPO 模型。
AWQ 和 vLLM 使用合并后的完整模型目录。

配置: 优先读取 config/config.local.yaml，缺失时读取 config/config.yaml
"""

import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer

from grid_llm.stage2_grpo.reward import build_deepseek_reward_func
from grid_llm.utils.common import (
    load_config, setup_logging, timer, extract_answer,
    SYSTEM_PROMPT, apply_chat_template,
)

logger = setup_logging("grpo")


def get_grpo_merged_dir(config: dict) -> str:
    """获取 GRPO 合并模型目录，兼容旧配置字段。"""
    grpo_cfg = config["grpo"]
    return grpo_cfg.get("merged_dir") or grpo_cfg["model_dir"]


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
    sft_cfg = config["sft"]
    output_dir = Path(grpo_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = config["sft"]["merged_dir"]

    # 加载数据
    ds, prompt_to_gt = load_grpo_data(config["data"]["grpo_eval_file"])

    # DeepSeek API 裁判
    reward_cfg = grpo_cfg["reward"]
    api_key = reward_cfg["judge_api_key"]
    if not api_key or "your-" in api_key:
        raise ValueError(
            "DeepSeek API key 未配置。请填写 config/config.local.yaml 的 grpo.reward.judge_api_key"
        )
    judge_model = reward_cfg.get("judge_model", "deepseek-chat")
    timeout = reward_cfg.get("judge_timeout", 30)
    logger.info(f"DeepSeek 裁判模型: {judge_model}")
    reward_func = build_deepseek_reward_func(api_key, prompt_to_gt, judge_model, timeout)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4-bit QLoRA 加载模型（省显存，32GB 单卡可训）
    logger.info(f"加载 SFT 模型 (4-bit): {model_path}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=sft_cfg["quantization"]["load_in_4bit"],
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=sft_cfg["quantization"]["bnb_4bit_use_double_quant"],
        bnb_4bit_quant_type=sft_cfg["quantization"]["bnb_4bit_quant_type"],
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)

    # LoRA 注入
    lora_cfg = sft_cfg["lora"]
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

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

    logger.info(f"开始 GRPO 训练 (LoRA, num_generations={train_cfg['num_generations']}, "
                f"beta={train_cfg['beta']}, lr={train_cfg['learning_rate']})")

    with timer("GRPO 训练", logger):
        trainer.train()

    # 保存 GRPO LoRA adapter
    adapter_dir = Path(grpo_cfg.get("adapter_dir", str(output_dir / "grpo_lora_adapter")))
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    logger.info(f"GRPO LoRA adapter 已保存到 {adapter_dir}")

    # 释放 4-bit 训练模型，再用 FP16/BF16 基座合并 LoRA，给 AWQ/vLLM 使用
    del trainer
    del model
    torch.cuda.empty_cache()

    merged_dir = Path(get_grpo_merged_dir(config))
    merged_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"加载 SFT 合并模型用于合并 GRPO LoRA: {model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    logger.info(f"加载 GRPO LoRA adapter: {adapter_dir}")
    merged_model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    logger.info("合并 GRPO LoRA 权重 ...")
    with timer("GRPO LoRA 合并", logger):
        merged_model = merged_model.merge_and_unload()

    logger.info(f"保存 GRPO 合并模型到 {merged_dir}")
    merged_model.save_pretrained(str(merged_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(merged_dir))
    logger.info("GRPO 合并模型保存完成，可用于 AWQ 量化和 vLLM 部署")

    mem = torch.cuda.max_memory_allocated() / 1024**3
    logger.info(f"训练完成，峰值显存: {mem:.1f} GB")


if __name__ == "__main__":
    main()
