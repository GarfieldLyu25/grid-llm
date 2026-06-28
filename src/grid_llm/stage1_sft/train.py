"""Stage 1: QLoRA SFT 微调。

对 Qwen2.5-7B-Instruct 做 4-bit QLoRA 增量训练，
让模型学会按「逐步推理 → 最终答案」格式输出数学解答。

产出: stage1_sft/output/lora_adapter/
"""

import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from grid_llm.utils.common import load_config, setup_logging, timer, SYSTEM_PROMPT

logger = setup_logging("sft")

# Chat template 包装：把 instruction/output 转成 Qwen 对话格式
CHAT_TEMPLATE = (
    "<|im_start|>system\n"
    "{system_prompt}<|im_end|>\n"
    "<|im_start|>user\n"
    "{instruction}<|im_end|>\n"
    "<|im_start|>assistant\n"
    "{output}<|im_end|>"
)


def load_dataset_from_jsonl(file_path: str) -> Dataset:
    """加载 JSONL 并转为 HuggingFace Dataset，应用 chat template。"""
    samples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    formatted = []
    for s in samples:
        text = CHAT_TEMPLATE.format(
            system_prompt=SYSTEM_PROMPT,
            instruction=s["instruction"],
            output=s["output"],
        )
        formatted.append({"text": text})

    return Dataset.from_list(formatted)


def main():
    config = load_config()
    sft_cfg = config["sft"]
    output_dir = Path(sft_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("加载 tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["name"],
        trust_remote_code=config["model"]["trust_remote_code"],
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("加载数据集 ...")
    train_ds = load_dataset_from_jsonl(config["data"]["train_file"])
    eval_ds = load_dataset_from_jsonl(config["data"]["eval_file"])
    logger.info(f"train: {len(train_ds)} 条, eval: {len(eval_ds)} 条")

    # Tokenize
    def tokenize_fn(examples):
        tokens = tokenizer(
            examples["text"],
            truncation=True,
            max_length=sft_cfg["training"]["max_seq_length"],
            padding=False,
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    logger.info("Tokenizing ...")
    train_ds = train_ds.map(tokenize_fn, remove_columns=["text"])
    eval_ds = eval_ds.map(tokenize_fn, remove_columns=["text"])

    # Quantization config
    logger.info("配置 QLoRA 4-bit 量化 ...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=sft_cfg["quantization"]["load_in_4bit"],
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=sft_cfg["quantization"]["bnb_4bit_use_double_quant"],
        bnb_4bit_quant_type=sft_cfg["quantization"]["bnb_4bit_quant_type"],
    )

    logger.info(f"加载基座模型: {config['model']['name']} ...")
    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["name"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=config["model"]["trust_remote_code"],
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)

    # LoRA config
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
    logger.info("LoRA 注入完成")
    model.print_trainable_parameters()

    # Training
    train_cfg = sft_cfg["training"]
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        num_train_epochs=train_cfg["num_epochs"],
        warmup_ratio=train_cfg["warmup_ratio"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        logging_steps=train_cfg["logging_steps"],
        save_steps=train_cfg["save_steps"],
        save_total_limit=train_cfg["save_total_limit"],
        bf16=train_cfg["bf16"],
        optim=train_cfg["optim"],
        seed=train_cfg["seed"],
        evaluation_strategy="steps",
        eval_steps=train_cfg["save_steps"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="tensorboard",
        run_name="grid-llm-sft",
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
    )

    # 检查 checkpoint
    checkpoint = None
    checkpoints = sorted(output_dir.glob("checkpoint-*"))
    if checkpoints:
        checkpoint = str(checkpoints[-1])
        logger.info(f"发现 checkpoint，从 {checkpoint} 续训")

    logger.info("开始 SFT 训练 ...")
    with timer("SFT 训练", logger):
        trainer.train(resume_from_checkpoint=checkpoint)

    # 保存 adapter
    adapter_dir = Path(sft_cfg["lora_adapter_dir"])
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    logger.info(f"LoRA adapter 已保存到 {adapter_dir}")

    # 打印显存信息
    mem = torch.cuda.max_memory_allocated() / 1024**3
    logger.info(f"训练完成，峰值显存: {mem:.1f} GB")


if __name__ == "__main__":
    main()
