"""
KnowFlow Chunk Splitter
=======================
RAG 第一阶段：把一篇笔记切成可检索、可溯源的 chunks。

设计原则
--------
1. 段落优先：按 \\n\\n（空行）切，符合小红书 / 公众号 / 博客的天然结构
2. 滑窗兜底：单段超过 MAX_CHUNK，按句子边界滑窗切分
3. 元数据齐全：每个 chunk 记录 char_start / char_end / chunk_type / section
4. 纯本地实现：0 API 调用，可在离线环境跑
5. 可溯源：char_start / char_end 是详情页跳转到原文位置的关键

切分规则速查
-----------
切点优先级（从高到低）：
    1. 空行 \\\\n\\\\s*\\\\n       → 段落分隔
    2. 标题 # / ## / ###     → section = "标题"
    3. 列表项 - 或 数字.       → 合并到同一段（不强切）
    4. 长段落内的句子 . ! ? \\n → 滑窗切分点

过滤：
    - 长度 < MIN_CHUNK（20字）丢弃（避免"未完待续"等噪声）
    - 全空白 / 全 emoji 丢弃

输入
----
split(text: str) -> list[Chunk]

注意：split 只处理 body，不处理 title。
调用方需要：
    1. 先把 title 作为一个 heading chunk 插入到结果最前面
    2. 再用 body 调 split，拿到正文 chunks
    3. 调整每个 chunk 的 char_start = title_len + 2 + chunk.char_start
    4. 用 insert_with_title() 这个 helper 一行搞定

返回
----
list[Chunk]，每个 Chunk 包含：
    text:        str   # 实际内容
    type:        str   # paragraph | heading | list | windowed
    section:     str   # 标题 | 正文
    char_start:  int   # 在原文中的字符偏移（含 title）
    char_end:    int   # 结束位置（不含）
    index:       int   # 在本文中的序号（从 0 开始）
"""

from __future__ import annotations
import re
from dataclasses import dataclass, asdict
from typing import List, Optional


# ---------- 默认参数 ----------
MIN_CHUNK = 20        # 短于 20 字丢弃
MAX_CHUNK = 400       # 超过 400 字走滑窗
WINDOW_SIZE = 256     # 滑窗大小
WINDOW_OVERLAP = 50   # 滑窗重叠


# ---------- 数据结构 ----------
@dataclass
class Chunk:
    """切分后的最小单元，所有字段都是溯源 / 检索 / UI 高亮的基础。"""
    text: str
    type: str           # paragraph | heading | list | windowed
    section: str        # 标题 | 正文
    char_start: int     # 在原文中（含 title）的起始偏移
    char_end: int       # 结束偏移
    index: int          # 在本文中的序号

    def to_dict(self) -> dict:
        return asdict(self)


# ---------- 工具函数 ----------
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
_LIST_BULLET_RE = re.compile(r"^[\-\*•]\s+", re.MULTILINE)
_LIST_ORDERED_RE = re.compile(r"^\d+[.\)、]\s+", re.MULTILINE)
# Unicode 圆圈数字：① ② ③ 等
_LIST_UNICODE_RE = re.compile(r"^[①②③④⑤⑥⑦⑧⑨⑩]\s*")
_LIST_ANY_RE = re.compile(
    r"^([\-\*•]\s+|\d+[.\)、]\s+|[①②③④⑤⑥⑦⑧⑨⑩]\s*)"
)
_SENTENCE_END_RE = re.compile(r"(?<=[。！？!?\n])")
_NOISE_RE = re.compile(r"^[\s\W_]+$", re.UNICODE)
# 短小标题候选：长度 < MIN_CHUNK 但有"标题感"（纯文本、不带句号）
_SHORT_HEADING_MAX = 20


def _is_noise(text: str) -> bool:
    """短段落 / 纯装饰段落判定。

    规则：
    - 全空白 / 全装饰：丢弃
    - 长度 < MIN_CHUNK 且无标点：默认丢弃（_is_short_heading 会捞起"3 样必点"类）
    - 长度 < MIN_CHUNK 但有完整句末标点（。.!?！？）：保留（短句也算内容）
    - 长度 >= MIN_CHUNK：保留
    """
    t = text.strip()
    if not t:
        return True
    if _NOISE_RE.match(t):
        return True
    has_text = any(0x0020 <= ord(c) <= 0x9FFF for c in t)
    if not has_text:
        return True
    # 长度足够：保留
    if len(t) >= MIN_CHUNK:
        return False
    # 短 + 有句末标点 → 短句，保留
    if any(p in t for p in "。.!?！？;；"):
        return False
    # 短 + 无标点 → 默认噪声（_is_short_heading 在外层救场）
    return True


_NOISE_WORDS = frozenset([
    "未完待续", "待续", "未完", "to be continued",
    "to be continue", "continued", "tbc", "TBC",
    "————", "———", "— — —", "...",
])

def _is_short_heading(para: str) -> bool:
    """短文本但具有标题感（无句末标点、纯文字、可能含数字/空格）。"""
    t = para.strip()
    if not t or len(t) > _SHORT_HEADING_MAX:
        return False
    # 显式噪声词
    if t in _NOISE_WORDS or t.replace(" ", "") in _NOISE_WORDS:
        return False
    # 含句末标点就不是标题
    if any(p in t for p in "。.!?！？;；\n"):
        return False
    # 标题不能以列表项标记开头
    if _LIST_BULLET_RE.match(t) or _LIST_ORDERED_RE.match(t) or _LIST_UNICODE_RE.match(t):
        return False
    return True


def _split_into_paragraphs(text: str) -> List[tuple[str, int, int]]:
    """
    按空行切段落，返回 [(content, start, end), ...]
    start / end 是每个段落在原文中的字符偏移。

    实现说明：手动扫描而不是依赖 re.split + 累加器，
    避免换行符序列的真实长度被错误估计。
    """
    paragraphs: List[tuple[str, int, int]] = []
    pattern = re.compile(r"\n\s*\n")

    cursor = 0
    for m in pattern.finditer(text):
        start, end = m.span()
        para_text = text[cursor:start]
        if para_text.strip():
            paragraphs.append((para_text, cursor, start))
        cursor = end
    # 最后一段
    if cursor < len(text):
        last = text[cursor:]
        if last.strip():
            paragraphs.append((last, cursor, len(text)))
    return paragraphs


def _is_heading(para: str) -> bool:
    """段内首行（忽略前导空行）以 # / ## / ### 开头。"""
    stripped = para.lstrip()
    if not stripped:
        return False
    first_line = stripped.split("\n", 1)[0]
    return bool(_HEADING_RE.match(first_line))


def _is_list_paragraph(para: str) -> bool:
    """整段都是列表项 → 视作列表。"""
    lines = [ln for ln in para.split("\n") if ln.strip()]
    if not lines:
        return False
    list_lines = [
        ln for ln in lines
        if _LIST_BULLET_RE.match(ln)
        or _LIST_ORDERED_RE.match(ln)
        or _LIST_UNICODE_RE.match(ln)
    ]
    return len(list_lines) >= max(1, len(lines) * 0.5)


def _window_long_paragraph(para: str, para_start: int) -> List[tuple[str, int, int]]:
    """
    长段落按句子滑窗，返回 (text, start, end)。
    start / end 是相对原文中 char_start 算的。
    """
    # 拆句子（保留分隔符）
    sentences = _SENTENCE_END_RE.split(para)
    sentences = [s for s in sentences if s]

    chunks: List[tuple[str, int, int]] = []
    if not sentences:
        return chunks

    window = ""
    win_start = 0
    cursor = 0

    for sent in sentences:
        if len(window) + len(sent) > WINDOW_SIZE and window:
            win_end = win_start + len(window)
            chunks.append((window, para_start + win_start, para_start + win_end))
            # 保留 overlap
            keep = window[-WINDOW_OVERLAP:] if len(window) > WINDOW_OVERLAP else window
            window = keep + sent
            win_start = win_end - len(keep)
        else:
            if not window:
                win_start = cursor
            window += sent
        cursor += len(sent)

    # 收尾
    if window:
        win_end = win_start + len(window)
        chunks.append((window, para_start + win_start, para_start + win_end))

    return chunks


# ---------- 主入口 ----------
def split(
    text: str,
    *,
    min_chunk: int = MIN_CHUNK,
    max_chunk: int = MAX_CHUNK,
    window_size: int = WINDOW_SIZE,
    window_overlap: int = WINDOW_OVERLAP,
) -> List[Chunk]:
    """
    把 body 文本切成 chunks。title 单独处理，见 insert_with_title()。

    Parameters
    ----------
    text : str
        笔记正文（不含 title）
    min_chunk / max_chunk / window_size / window_overlap :
        可覆盖默认值，主要给测试用

    Returns
    -------
    List[Chunk]
    """
    # 重设模块级参数（仅本次调用内有效）
    global MIN_CHUNK, MAX_CHUNK, WINDOW_SIZE, WINDOW_OVERLAP
    _saved = (MIN_CHUNK, MAX_CHUNK, WINDOW_SIZE, WINDOW_OVERLAP)
    MIN_CHUNK, MAX_CHUNK, WINDOW_SIZE, WINDOW_OVERLAP = (
        min_chunk, max_chunk, window_size, window_overlap,
    )
    try:
        return _split_impl(text)
    finally:
        MIN_CHUNK, MAX_CHUNK, WINDOW_SIZE, WINDOW_OVERLAP = _saved


def _split_impl(text: str) -> List[Chunk]:
    chunks: List[Chunk] = []

    paragraphs = _split_into_paragraphs(text)

    for para, para_start, para_end in paragraphs:
        # 优先级 0: 全空白 / 纯装饰噪声
        if _is_pure_noise(para):
            continue

        # 优先级 1: 短标题（如 "3 样必点"）— 在 _is_noise 之前判定
        # 短+无标点的合法小标题不能被当噪声丢
        if _is_short_heading(para):
            chunks.append(Chunk(
                text=para.strip(),
                type="heading",
                section="正文",
                char_start=para_start,
                char_end=para_end,
                index=len(chunks),
            ))
            continue

        # 优先级 2: 标准噪声（短+无标点 / 短+有标点但太短等）
        if _is_noise(para):
            continue

        # 优先级 3: Markdown # / ## 标题
        if _is_heading(para):
            chunks.append(Chunk(
                text=para.strip(),
                type="heading",
                section="标题",
                char_start=para_start,
                char_end=para_end,
                index=len(chunks),
            ))
            continue

        # 优先级 4: 列表段
        if _is_list_paragraph(para):
            chunks.append(Chunk(
                text=para.strip(),
                type="list",
                section="正文",
                char_start=para_start,
                char_end=para_end,
                index=len(chunks),
            ))
            continue

        # 优先级 5: 短段落
        if len(para.strip()) <= MAX_CHUNK:
            chunks.append(Chunk(
                text=para.strip(),
                type="paragraph",
                section="正文",
                char_start=para_start,
                char_end=para_end,
                index=len(chunks),
            ))
            continue

        # 优先级 6: 长段落 → 滑窗
        windows = _window_long_paragraph(
            para.strip(),
            para_start=para_start,
        )
        for wtext, wstart, wend in windows:
            chunks.append(Chunk(
                text=wtext,
                type="windowed",
                section="正文",
                char_start=wstart,
                char_end=wend,
                index=len(chunks),
            ))

    return chunks


def _is_pure_noise(text: str) -> bool:
    """绝对噪声：全空白 / 全装饰 / 显式噪声词。"""
    t = text.strip()
    if not t:
        return True
    if t in _NOISE_WORDS or t.replace(" ", "") in _NOISE_WORDS:
        return True
    if _NOISE_RE.match(t):
        return True
    has_text = any(0x0020 <= ord(c) <= 0x9FFF for c in t)
    if not has_text:
        return True
    return False


def insert_with_title(title: str, body_chunks: List[Chunk]) -> List[Chunk]:
    """
    把 title 作为第一个 heading chunk 插入，并修正 body_chunks 的 char_start。

    偏移约定：原文中 "title\\n\\nbody" 的字符位置。
    """
    if not title:
        # 没有 title 就用 body chunks 自身（重新分配 index）
        for i, c in enumerate(body_chunks):
            c.index = i
        return body_chunks

    title_clean = title.strip()
    offset = len(title) + 2  # title + \n\n 分隔

    # 修正 body chunks 的偏移
    new_body: List[Chunk] = []
    for c in body_chunks:
        new_body.append(Chunk(
            text=c.text,
            type=c.type,
            section=c.section,
            char_start=c.char_start + offset,
            char_end=c.char_end + offset,
            index=c.index + 1,  # 让出 index 0 给 title
        ))

    title_chunk = Chunk(
        text=title_clean,
        type="heading",
        section="标题",
        char_start=0,
        char_end=len(title_clean),
        index=0,
    )
    return [title_chunk] + new_body


# ---------- CLI / 自检 ----------
if __name__ == "__main__":
    body = """路过愚园路偶然发现的小店，门面非常不起眼，但推门进去别有洞天。

3 样必点

① 味噌烤银鳕鱼：外皮微焦内里嫩滑，咸甜平衡。
② 手作玉子烧：现点现做等 15 分钟，但是值得。
③ 柚子盐拉面：汤底清爽不腻，柚子香气很正。

避雷明太鱼籽茶泡饭，太咸。"""
    title = "上海宝藏日料店｜藏在愚园路的深夜食堂，3 样必点"

    body_chunks = split(body)
    result = insert_with_title(title, body_chunks)
    print(f"切出 {len(result)} 个 chunks:")
    for c in result:
        print(f"  [{c.index:02d}] {c.type:9s} | {c.section} | "
              f"[{c.char_start:3d}:{c.char_end:3d}] | {c.text[:30]}...")
