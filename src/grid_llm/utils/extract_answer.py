"""extract_answer 独立模块，供 GRPO reward 函数独立使用。"""

from grid_llm.utils.common import extract_answer, normalize_answer, answers_equal

__all__ = ["extract_answer", "normalize_answer", "answers_equal"]
