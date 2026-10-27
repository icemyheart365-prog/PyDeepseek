# pip install tokenizers
# 用法:
#   from token_counter import token_counter
#   n = token_counter("要统计的文本")
#
# 可选接口（供主脚本回退用）：
#   try_load_token_counter()  返回 token_counter 或 None
#   approximate_token_count() 无论 tokenizers 是否可用都能返回一个整数
import os

try:
    from tokenizers import Tokenizer
except ImportError:  # 例如 A-Shell 等装不上 tokenizers 的环境
    Tokenizer = None

_TOKENIZER = None
_LOAD_ERROR = None

if Tokenizer is not None:
    try:
        _TOKENIZER = Tokenizer.from_file(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "tokenizer.json")
        )
    except Exception as exc:  # 文件缺失/损坏时由 tokenizers 抛出底层异常
        _LOAD_ERROR = exc


def token_counter(text: str) -> int:
    """返回文本的 token 数量。"""
    if _TOKENIZER is None:
        raise RuntimeError(f"tokenizer 不可用：{_LOAD_ERROR or '未安装 tokenizers'}")
    return len(_TOKENIZER.encode(text).ids)


def approximate_token_count(text: str) -> int:
    """无法加载 tokenizer 时，用 UTF-8 字节数粗略回退（约 4 字节 1 token）。"""
    return max(1, (len(text.encode("utf-8")) + 3) // 4)


def try_load_token_counter():
    """tokenizer 可用时返回 token_counter，否则返回 None 并保留原因。"""
    if _TOKENIZER is None:
        return None
    return token_counter


if __name__ == "__main__":
    print(token_counter(input()))
