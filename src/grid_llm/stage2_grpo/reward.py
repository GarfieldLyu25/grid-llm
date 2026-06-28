"""GRPO 数学奖励函数。

对每组生成的 completion，评估数学正确性。
- 答案正确 + 有推理步骤 → 1.0
- 答案正确但无推理 → 0.9
- 答案错误 → 0.0

用法（单元测试）:
  uv run python src/grid_llm/stage2_grpo/reward.py --test
"""

import sys
import logging

from grid_llm.utils.common import extract_answer, answers_equal

logger = logging.getLogger("grid-llm.reward")


def math_reward_with_answer(
    completions: list[str],
    ground_truths: list[str],
) -> list[float]:
    """带标准答案的奖励函数。

    用作 GRPOTrainer reward_funcs 的闭包包装。

    评分逻辑:
    - 答案完全匹配 + 有推理步骤 → 1.0
    - 答案完全匹配 + 无推理步骤 → 0.9
    - 答案不匹配 → 0.0
    """
    scores = []
    reasoning_keywords = ["步骤", "推理", "分析", "计算", "过程", "思路", "1.", "2.", "步骤1"]

    for completion, gt in zip(completions, ground_truths):
        answer = extract_answer(completion)
        is_correct = answer is not None and answers_equal(answer, gt)
        has_reasoning = any(kw in completion for kw in reasoning_keywords)

        if is_correct and has_reasoning:
            scores.append(1.0)
        elif is_correct and not has_reasoning:
            scores.append(0.9)
        else:
            scores.append(0.0)

    return scores


def build_reward_func(prompt_to_gt: dict[str, str]):
    """构建 GRPOTrainer 兼容的奖励函数闭包。

    prompt_to_gt: {prompt_text: ground_truth_answer}

    返回的 reward_func 接收 (completions, prompts, **kwargs)，
    通过 prompts 参数查找对应的 ground_truth 做正确匹配。
    """
    def reward_func(
        completions: list[str],
        prompts: list[str] | None = None,
        **kwargs,
    ) -> list[float]:
        gts = []
        if prompts:
            for p in prompts:
                gts.append(prompt_to_gt.get(p, "0"))
        else:
            # 兜底：无 prompts 参数时用零填充
            gts = ["0"] * len(completions)
        return math_reward_with_answer(completions, gts)

    return reward_func


# ---- 单元测试 ----

def test_reward():
    """验证奖励函数行为。"""
    logger.info("=== 奖励函数单元测试 ===")

    # 完整正确（答案对+有推理）
    good = """逐步推理：
1. 鸡 + 兔 = 35
2. 2×鸡 + 4×兔 = 94
3. 兔 = (94 - 2×35) / 2 = 12
4. 鸡 = 35 - 12 = 23

最终答案：鸡有23只，兔有12只"""
    scores = math_reward_with_answer(completions=[good], ground_truths=["23"])
    logger.info(f"完整正确: {scores[0]} (期望 1.0)")
    assert scores[0] == 1.0, f"期望 1.0, 实际 {scores[0]}"

    # 答案对但无推理
    no_reasoning = "答案是 23 只鸡，12 只兔。"
    scores = math_reward_with_answer(completions=[no_reasoning], ground_truths=["23"])
    logger.info(f"答案对无推理: {scores[0]} (期望 0.9)")
    assert scores[0] == 0.9, f"期望 0.9, 实际 {scores[0]}"

    # 答案错误
    wrong = """逐步推理：
1. 鸡 + 兔 = 35
2. 2×鸡 + 4×兔 = 94
3. 兔 = 88, 鸡 = -53

最终答案：鸡有-53只，兔有88只"""
    scores = math_reward_with_answer(completions=[wrong], ground_truths=["23"])
    logger.info(f"答案错误: {scores[0]} (期望 0.0)")
    assert scores[0] == 0.0, f"期望 0.0, 实际 {scores[0]}"

    logger.info("=== 全部测试通过 ===")


if __name__ == "__main__":
    from grid_llm.utils.common import setup_logging
    setup_logging("grid-llm")

    if "--test" in sys.argv:
        test_reward()
    else:
        logger.info("用法: uv run python src/grid_llm/stage2_grpo/reward.py --test")
