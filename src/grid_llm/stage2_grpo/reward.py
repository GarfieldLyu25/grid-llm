"""GRPO 数学奖励函数 v3 — 诚实版。

规则匹配无法判断推理质量。不假装能。
只判两样确定的事：
  - 答案对不对（extract_answer + answers_equal）
  - 格式像不像正经答题（有没有答案标记、够不够长）

评分:
  - 答案对 + 格式完整 → 1.0
  - 答案对 + 没有格式 → 0.5（蒙对的，给一半）
  - 答案错 → 0.0

用法:
  uv run python src/grid_llm/stage2_grpo/reward.py --test
"""

import sys
import logging

from grid_llm.utils.common import extract_answer, answers_equal

logger = logging.getLogger("grid-llm.reward")

# ── 格式常量 ──────────────────────────────────────────

_MIN_TOTAL_CHARS = 30        # 整个输出最少字符数（低于这个基本是裸答案）
_ANSWER_MARKERS = ["最终答案", "答案：", "答案:"]


def _looks_like_answer(completion: str) -> bool:
    """检查输出是否「看起来像正经答题」。

    只做最基本的格式判断，不假装能评估推理质量：
    - 有明确的答案标记
    - 不是太短（短说明没过程，就是蒙）
    """
    text = completion.strip()
    has_marker = any(m in text for m in _ANSWER_MARKERS)
    long_enough = len(text) >= _MIN_TOTAL_CHARS
    return has_marker and long_enough


# ── 主评分函数 ────────────────────────────────────────

def math_reward_with_answer(
    completions: list[str],
    ground_truths: list[str],
) -> list[float]:
    """诚实版数学奖励函数。

    不假装能判推理质量，只判：
    - 答案正确 + 输出像正经答题 → 1.0
    - 答案正确 + 输出不像正经答题 → 0.5（蒙的，给一半信号）
    - 答案错误 → 0.0
    """
    scores = []

    for completion, gt in zip(completions, ground_truths):
        answer = extract_answer(completion)
        is_correct = answer is not None and answers_equal(answer, gt)
        has_format = _looks_like_answer(completion)

        if is_correct and has_format:
            scores.append(1.0)
        elif is_correct and not has_format:
            scores.append(0.5)
        else:
            scores.append(0.0)

    return scores


def build_reward_func(prompt_to_gt: dict[str, str]):
    """构建 GRPOTrainer 兼容的规则版奖励函数闭包。

    prompt_to_gt: {prompt_text: ground_truth_answer}
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
            gts = ["0"] * len(completions)
        return math_reward_with_answer(completions, gts)

    return reward_func


# ── DeepSeek API 裁判 ──────────────────────────────────

_JUDGE_PROMPT = """你是一个数学老师，请批改学生的解答。

题目：{prompt}
标准答案：{ground_truth}

学生的解答：
{completion}

请从以下几方面评分（0.0 ~ 1.0）：
1. 最终答案是否正确（占 60%）
2. 推理过程是否合理、步骤是否清晰（占 30%）
3. 解答格式是否规范（占 10%）

只回答一个数字（如 0.85），不要解释。"""


def _judge_via_deepseek(
    api_key: str,
    prompt: str,
    ground_truth: str,
    completion: str,
    model: str = "deepseek-chat",
    timeout: int = 30,
) -> float:
    """调用 DeepSeek API 评判一条回答。"""
    import requests

    judge_text = _JUDGE_PROMPT.format(
        prompt=prompt[:500],
        ground_truth=ground_truth,
        completion=completion[:1000],
    )

    try:
        resp = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "你是一个严格的数学老师，只回答数字。"},
                    {"role": "user", "content": judge_text},
                ],
                "max_tokens": 10,
                "temperature": 0.0,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()

        import re
        match = re.search(r"(\d+\.?\d*)", content)
        if match:
            return min(max(float(match.group(1)), 0.0), 1.0)
        return 0.0

    except Exception as e:
        logger.warning(f"DeepSeek API 调用失败: {e}，该条给 0.0")
        return 0.0


def math_reward_with_deepseek(
    api_key: str,
    prompt_to_gt: dict[str, str],
    completions: list[str],
    prompts: list[str],
) -> list[float]:
    """DeepSeek API 版奖励函数 — 真能看推理质量。

    对每条 completion 调 DeepSeek，返回 0~1 连续分数。
    """
    scores = []
    for completion, prompt in zip(completions, prompts):
        gt = prompt_to_gt.get(prompt, "0")
        score = _judge_via_deepseek(api_key, prompt, gt, completion)
        scores.append(score)
    return scores


def build_deepseek_reward_func(
    api_key: str,
    prompt_to_gt: dict[str, str],
    model: str = "deepseek-chat",
    timeout: int = 30,
):
    """构建 DeepSeek API 版奖励函数闭包。

    评分维度：答案正确性(60%) + 推理质量(30%) + 格式规范(10%)
    返回 0~1 连续分数，比规则版细得多。
    """
    def reward_func(
        completions: list[str],
        prompts: list[str] | None = None,
        **kwargs,
    ) -> list[float]:
        if not prompts:
            prompts = [""] * len(completions)
        scores = []
        for completion, prompt in zip(completions, prompts):
            gt = prompt_to_gt.get(prompt, "0")
            score = _judge_via_deepseek(api_key, prompt, gt, completion, model, timeout)
            scores.append(score)
        return scores

    return reward_func


# ── 单元测试 ──────────────────────────────────────────

def test_reward():
    """验证诚实版奖励函数。"""
    logger.info("=== 奖励函数 v3 诚实版 单元测试 ===")

    gt = ["23"]

    # 1) 答案对 + 格式完整 → 1.0
    good = """逐步推理：
1. 设鸡有 x 只，兔有 y 只
2. x + y = 35，2x + 4y = 94
3. 解得 y = 12，x = 23
最终答案：鸡有 23 只，兔有 12 只。"""
    s = math_reward_with_answer([good], gt)[0]
    logger.info(f"答案对+格式完整: {s} (期望 1.0)")
    assert s == 1.0, f"期望 1.0，实际 {s}"

    # 2) 答案对 + 没有格式 → 0.5
    bare = "23"
    s = math_reward_with_answer([bare], gt)[0]
    logger.info(f"裸答案: {s} (期望 0.5)")
    assert s == 0.5, f"期望 0.5，实际 {s}"

    # 3) 答案错 → 0.0（不管格式怎么样）
    wrong = """逐步推理：
1. 计算 94÷2 = 47 只鸡
2. 兔 = 35-47 = -12
最终答案：鸡有 47 只，兔有 -12 只。"""
    s = math_reward_with_answer([wrong], gt)[0]
    logger.info(f"答案错: {s} (期望 0.0)")
    assert s == 0.0, f"期望 0.0，实际 {s}"

    # 4) 答案对但格式不完整（有标记但太短）→ 0.5
    short = "最终答案：23"
    s = math_reward_with_answer([short], gt)[0]
    logger.info(f"答案对格式不完整: {s} (期望 0.5)")
    assert s == 0.5, f"期望 0.5，实际 {s}"

    # 5) 纯噪音 → 0.0
    noise = "asdfgh"
    s = math_reward_with_answer([noise], gt)[0]
    logger.info(f"纯噪音: {s} (期望 0.0)")
    assert s == 0.0, f"期望 0.0，实际 {s}"

    logger.info("=== 全部测试通过 ===")


if __name__ == "__main__":
    from grid_llm.utils.common import setup_logging
    setup_logging("grid-llm")

    if "--test" in sys.argv:
        test_reward()
    else:
        logger.info("用法: uv run python src/grid_llm/stage2_grpo/reward.py --test")
