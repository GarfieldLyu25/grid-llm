"""公共工具：配置加载、日志、计时、答案提取、共享 Prompt。"""

import os
import re
import sys
import time
import logging
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime

import yaml

DEFAULT_CONFIG_PATH = Path("config/config.yaml")
LOCAL_CONFIG_PATH = Path("config/config.local.yaml")


# ---- 日志（控制台 + 文件双写） ----

_LOG_INITIALIZED = False


def setup_logging(
    name: str = "grid-llm",
    level: int = logging.INFO,
    log_dir: str = "logs",
) -> logging.Logger:
    """统一日志：同时输出到控制台和 logs/ 目录。

    日志文件按日期分目录: logs/2026-06-28/grid-llm.log
    控制台: INFO 级别；文件: DEBUG 级别（保留完整记录）。
    """
    global _LOG_INITIALIZED
    logger = logging.getLogger(name)

    if not _LOG_INITIALIZED:
        logger.setLevel(logging.DEBUG)
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)-5s %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # 控制台
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(fmt)
        console.setLevel(level)
        logger.addHandler(console)

        # 文件日志: logs/YYYY-MM-DD/name.log
        today = datetime.now().strftime("%Y-%m-%d")
        log_path = Path(log_dir) / today
        log_path.mkdir(parents=True, exist_ok=True)
        file_h = logging.FileHandler(log_path / f"{name}.log", encoding="utf-8")
        file_h.setFormatter(fmt)
        file_h.setLevel(logging.DEBUG)
        logger.addHandler(file_h)

        _LOG_INITIALIZED = True

    return logger


@contextmanager
def timer(description: str, logger: logging.Logger | None = None):
    """上下文计时器，自动记录耗时。"""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    msg = f"{description} — 耗时 {elapsed:.1f}s"
    if logger:
        logger.info(msg)
    else:
        logging.getLogger("grid-llm").info(msg)


# ---- 配置加载 ----

def default_config_path() -> Path:
    """返回默认配置路径：本地配置优先，仓库模板兜底。"""
    return LOCAL_CONFIG_PATH if LOCAL_CONFIG_PATH.exists() else DEFAULT_CONFIG_PATH


def _resolve_path(value: str, base_dir: str) -> str:
    """替换路径中的 {base_dir} 占位符。"""
    return value.replace("{base_dir}", base_dir)


def _resolve_config(obj, base_dir: str):
    """递归遍历配置，替换所有字符串中的 {base_dir}。"""
    if isinstance(obj, str):
        return _resolve_path(obj, base_dir)
    if isinstance(obj, dict):
        return {k: _resolve_config(v, base_dir) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_config(item, base_dir) for item in obj]
    return obj


def load_config(config_path: str | None = None) -> dict:
    """加载 YAML 配置文件，自动解析 {base_dir} 占位符。

    默认优先读取 config/config.local.yaml；如果不存在，再读取
    config/config.yaml。前者用于本机真实运行，后者作为仓库模板。
    """
    path = Path(config_path) if config_path else default_config_path()

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    base_dir = raw.get("base_dir", "./").rstrip("/")
    if not Path(base_dir).is_absolute():
        base_dir = str(Path(base_dir).resolve())
    os.makedirs(base_dir, exist_ok=True)
    return _resolve_config(raw, base_dir)


# ---- 答案提取与比对 ----

def extract_answer(text: str) -> str | None:
    """从模型输出中提取最终答案。

    优先级：
    1. 「最终答案：xxx」或「答案：xxx」
    2. 末尾最后一个数值
    """
    patterns = [
        r"最终答案[：:]\s*(.+?)(?:\n|$)",
        r"答案[：:]\s*(.+?)(?:\n|$)",
        r"答[：:]\s*(.+?)(?:\n|$)",
        r"所以[，,]?\s*答案是?\s*[：:]*\s*(.+?)(?:\n|$)",
    ]
    for pat in patterns:
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            return match.group(1).strip().rstrip(".。，")

    numbers = re.findall(r"-?\d+\.?\d*", text)
    if numbers:
        return numbers[-1]
    return None


def normalize_answer(answer: str) -> str:
    """归一化答案字符串：去单位、去标点、去空格。"""
    ans = answer.strip().lower()
    ans = re.sub(r"[个只条块元角分寸米千米公里吨千克克]", "", ans)
    return ans.rstrip(".。，,、;；")


def answers_equal(pred: str, target: str, tolerance: float = 1e-6) -> bool:
    """判断两个答案是否相等（数值容差、分数、中文数字）。"""
    import fractions

    pred = normalize_answer(pred)
    target = normalize_answer(target)

    if pred == target:
        return True

    try:
        p_val = float(pred.replace(",", ""))
        t_val = float(target.replace(",", ""))
        return abs(p_val - t_val) < tolerance
    except (ValueError, TypeError):
        pass

    try:
        p_frac = fractions.Fraction(pred)
        t_frac = fractions.Fraction(target)
        return abs(float(p_frac) - float(t_frac)) < tolerance
    except (ValueError, ZeroDivisionError):
        pass

    return False


# ---- 共享 Prompt 模板 ----

SYSTEM_PROMPT = "你是一个数学老师。对于每道数学题，请逐步推理，最后给出答案。"

# 数据过滤阈值
MIN_INSTRUCTION_LEN = 4
MAX_INSTRUCTION_LEN = 2000
MIN_OUTPUT_LEN = 4

# 推理默认值
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_TEMPERATURE = 0.0


def build_chat_messages(instruction: str, system_prompt: str | None = None) -> list[dict]:
    """构建标准对话消息。"""
    return [
        {"role": "system", "content": system_prompt or SYSTEM_PROMPT},
        {"role": "user", "content": instruction},
    ]


def apply_chat_template(tokenizer, instruction: str, add_generation_prompt: bool = True) -> str:
    """将 instruction 转为 chat template 文本。"""
    messages = build_chat_messages(instruction)
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt
    )
