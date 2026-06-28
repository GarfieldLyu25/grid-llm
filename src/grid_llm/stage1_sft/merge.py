"""合并 LoRA adapter 到基座模型，产出完整 SFT 模型。

产出: stage1_sft/output/sft_merged/
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from grid_llm.utils.common import load_config, setup_logging, timer

logger = setup_logging("merge")


def main():
    config = load_config()
    sft_cfg = config["sft"]
    adapter_dir = sft_cfg["lora_adapter_dir"]
    merged_dir = sft_cfg["merged_dir"]

    logger.info(f"加载基座模型: {config['model']['name']} ...")
    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["name"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    logger.info(f"加载 LoRA adapter: {adapter_dir} ...")
    model = PeftModel.from_pretrained(model, adapter_dir)

    logger.info("合并权重 ...")
    with timer("合并", logger):
        model = model.merge_and_unload()

    logger.info(f"保存合并模型到 {merged_dir} ...")
    model.save_pretrained(str(merged_dir))

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    tokenizer.save_pretrained(str(merged_dir))

    logger.info("合并完成！")


if __name__ == "__main__":
    main()
