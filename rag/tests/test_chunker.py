"""
KnowFlow Chunker Tests
======================
用 5 类真实场景验证切分器：

1. 短笔记（普通小红书探店文）— 应按段落切
2. 长段落（必须走滑窗）— 应按句子滑窗
3. 列表笔记（多重 - / 数字.）— 应识别为 list chunk
4. 标题 + 章节结构（# / ##）— heading chunk 在前
5. 噪声过滤（"未完待续" / 纯 emoji / 短句）— 应被丢弃

跑法：python3 rag/tests/test_chunker.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from chunker import split, insert_with_title, Chunk


def _split_with_title(body: str, title: str = "") -> list:
    """helper：等价于 split(body, title=title)"""
    return insert_with_title(title, split(body))


# ---------- 1. 短笔记 ----------
SAMPLE_SHORT = {
    "title": "上海宝藏日料店｜藏在愚园路的深夜食堂，3 样必点",
    "body": """路过愚园路偶然发现的小店，门面非常不起眼，但推门进去别有洞天。

3 样必点

① 味噌烤银鳕鱼：外皮微焦内里嫩滑，咸甜平衡。
② 手作玉子烧：现点现做等 15 分钟，但是值得。
③ 柚子盐拉面：汤底清爽不腻，柚子香气很正。

避雷明太鱼籽茶泡饭，太咸。""",
}


def test_short_note_basic():
    """短笔记：标题 + 4 段正文 + 1 列表。"""
    chunks = _split_with_title(SAMPLE_SHORT["body"], title=SAMPLE_SHORT["title"])
    # 1 heading(title) + 1 paragraph(开头) + 1 paragraph("3 样必点") + 1 list(①②③) + 1 paragraph(避雷)
    # 实际：title + "路过愚园路..." + "3 样必点" + 列表 + "避雷..." = 5 个
    assert len(chunks) == 5, f"expected 5 chunks, got {len(chunks)}: {[c.text[:20] for c in chunks]}"
    assert chunks[0].type == "heading"
    assert chunks[0].section == "标题"
    # 列表识别
    list_chunks = [c for c in chunks if c.type == "list"]
    assert len(list_chunks) == 1
    print(f"  ✓ test_short_note_basic: {len(chunks)} chunks, types = {[c.type for c in chunks]}")


def test_short_note_offset_monotonic():
    """chunk 偏移在原文中是单调递增的（溯源的前置条件）。"""
    chunks = _split_with_title(SAMPLE_SHORT["body"], title=SAMPLE_SHORT["title"])
    for i in range(1, len(chunks)):
        assert chunks[i].char_start >= chunks[i - 1].char_start, \
            f"offset not monotonic: chunk {i} start {chunks[i].char_start} < prev {chunks[i-1].char_start}"
    assert chunks[0].char_start == 0
    # 头尾偏移要在合理范围
    total_len = len(SAMPLE_SHORT["title"]) + 2 + len(SAMPLE_SHORT["body"])
    assert chunks[-1].char_end <= total_len + 5, \
        f"last chunk end {chunks[-1].char_end} > total {total_len}"
    print(f"  ✓ test_short_note_offset_monotonic: offsets valid [0, {chunks[-1].char_end}]")


def test_short_note_offset_can_locate_title():
    """根据 char_start / char_end 反推回原文应能定位正确。"""
    title = SAMPLE_SHORT["title"]
    body = SAMPLE_SHORT["body"]
    full_text = title + "\n\n" + body
    chunks = _split_with_title(body, title=title)
    for c in chunks:
        # 反推
        recovered = full_text[c.char_start:c.char_end].strip()
        assert recovered == c.text or recovered in c.text or c.text in recovered, \
            f"chunk {c.index}: recovered '{recovered}' != chunk text '{c.text}'"
    print(f"  ✓ test_short_note_offset_can_locate_title: all {len(chunks)} chunks round-trip OK")


# ---------- 2. 长段落 ----------
SAMPLE_LONG = {
    "title": "RAG 实战笔记：Chunk 切分的 5 种策略对比",
    "body": """在 RAG 系统里，Chunk 切分是整个 pipeline 中最容易被忽视、但影响最大的一环。切得太粗，检索粒度太粗、答案不精准；切得太细，又会丢上下文；切得不对齐语义边界，LLM 读起来会感到割裂；切得太规则，又会让某些内容长度失控。综合来看，切分策略的选择要兼顾语义完整、检索粒度、实现成本、调试难度四个维度。一种好的切分策略应该满足：chunk 内信息密度均匀，chunk 之间不重复太多，溯源时定位精确，导入时不要慢到用户受不了。

第一种是固定 token 切分，按 256 或 512 token 强行切一刀。实现最简单，但经常会切断语义，比如把一个完整的论点一分为二，让 LLM 读起来很费劲。在英文场景下还好，中文里 token 边界和字边界不一致，问题更明显。我们团队曾经在 1 万篇小红书笔记上做过实验，固定 256 token 切分的召回准确率只有 62%，明显低于段落切分。

第二种是句子切分，按中文的句号、问号、感叹号、英文的 .!? 切。能保住单句完整，但遇到长句就失效，一段可能只有 1 个 chunk，召回粒度还是太粗。而且对于小红书这种大量短句的笔记，句子切分会产出一堆 10-20 字的 chunk，浪费 embedding 调用。

第三种是段落切分，按 \\n\\n 切。这个最贴合笔记类内容，标题 / 分段 / 列表天然有结构。但段落长度不均，有的 30 字有的 800 字，超长段落还得兜底。在我们的实验里，段落切分 + 滑窗兜底能拿到 78% 的召回准确率，是综合最优解。

第四种是滑动窗口，前面三种的妥协方案。256 token 窗口 + 50 token 重叠，能兼顾上下文和粒度，是目前工业界的事实标准。缺点是 chunk 数量会多 30% 左右，存储成本略高，但换来的是稳定的检索质量。

第五种是语义切分，先用 embedding 算相邻段落的相似度，找到语义断点再切。效果最好，召回准确率能到 85% 以上，但要多调一次 embedding API，导入一篇笔记要慢 2-3 秒，不适合 MVP。等用户量起来、对效果有更高要求时再考虑。""",
}


def test_long_paragraph_triggers_window():
    """长段落（> max_chunk）必须走滑窗，产生 windowed chunk。
    测试时把 max_chunk 调小，强制触发滑窗逻辑。
    """
    # 用 max_chunk=150 强制触发：part 0 (220字) 会被切
    body_chunks = split(SAMPLE_LONG["body"], max_chunk=150, window_size=80, window_overlap=20)
    chunks = insert_with_title(SAMPLE_LONG["title"], body_chunks)
    types = [c.type for c in chunks]
    assert "windowed" in types, f"expected windowed chunks, got types={types}"
    n_windowed = sum(1 for c in chunks if c.type == "windowed")
    assert n_windowed >= 1, f"expected >= 1 windowed, got {n_windowed}"
    print(f"  ✓ test_long_paragraph_triggers_window: {n_windowed} windowed chunks out of {len(chunks)} total")


def test_long_paragraph_window_size():
    """滑窗大小必须遵守限制（个别边界除外）。"""
    body_chunks = split(SAMPLE_LONG["body"], max_chunk=150, window_size=80, window_overlap=20)
    chunks = insert_with_title(SAMPLE_LONG["title"], body_chunks)
    for c in chunks:
        if c.type == "windowed":
            # 窗口可能略大于 window_size（最后一段）
            assert len(c.text) <= 80 + 30, \
                f"windowed chunk too large: {len(c.text)} chars"
    print(f"  ✓ test_long_paragraph_window_size: all windowed chunks within size limit")


# ---------- 3. 列表笔记 ----------
SAMPLE_LIST = {
    "title": "敏感肌夏季护肤｜成分党整理的 6 款无雷单品",
    "body": """夏天敏感肌容易泛红闷痘，下面是 6 款我反复回购的单品。

- 洁面：珂润润浸保湿洁颜泡沫。氨基酸体系，挤出来就是泡沫，懒人友好。
- 精华：薇诺娜舒敏保湿特护精华。马齿苋提取物舒缓泛红，质地清爽。
- 面霜：理肤泉 B5 修复面霜。b5 + 积雪草，烂脸期救星，略油。
- 防晒：怡丽丝尔小金管。物化结合，成膜快不搓泥。
- 卸妆：Fancl 速净卸妆油。乳化快，不闷痘，无香料。
- 面膜：可复美重组胶原蛋白面膜。医美术后修复用，敏感肌一周 1-2 次即可。

以上 6 款覆盖洁面到面膜，可以闭眼入。""",
}


def test_list_paragraph_recognized():
    """整段都是 - / 数字. 列表项时，应识别为 list chunk 而不是 paragraph。"""
    chunks = _split_with_title(SAMPLE_LIST["body"], title=SAMPLE_LIST["title"])
    types = [c.type for c in chunks]
    assert "list" in types, f"expected list chunk, got types={types}"
    list_chunks = [c for c in chunks if c.type == "list"]
    assert any("珂润" in c.text for c in list_chunks)
    print(f"  ✓ test_list_paragraph_recognized: list chunks = {len(list_chunks)}")


# ---------- 4. 标题 + 章节 ----------
SAMPLE_HEADINGS = {
    "title": "京都 5 月自由行攻略",
    "body": """# 5 月为什么适合去京都

5 月是京都最舒服的季节，气温 18-25°C，既不冷也不热。新绿和樱花尾季叠加，人比 4 月少一半。

# 行程推荐

## Day 1：岚山 + 渡月桥

清晨 6 点去岚山竹林，游客还没进来，随便拍都是大片。渡月桥边上有家豆腐料理店必去。

## Day 2：银阁寺 + 哲学之道

银阁寺清晨 7 点开门，哲学之道沿途 3 家町家咖啡，最推荐 % Arabica Kyoto。

# 避雷提醒

- 5 月中旬有葵祭，会封路
- 餐厅基本都收 10% 服务费
- JR 京都站到市区用 ICOCA 卡最方便""",
}


def test_headings_and_sections():
    """# / ## 标题应被识别为 heading chunk。"""
    chunks = _split_with_title(SAMPLE_HEADINGS["body"], title=SAMPLE_HEADINGS["title"])
    headings = [c for c in chunks if c.type == "heading"]
    # 1 主标题(title) + 3 个 # + 2 个 ## = 6
    assert len(headings) >= 5, f"expected >= 5 headings, got {len(headings)}"
    # title chunk section="标题"，正文中识别出的 heading section="正文"
    sections = {h.section for h in headings}
    assert "标题" in sections, f"expected 标题 section in {sections}"
    print(f"  ✓ test_headings_and_sections: {len(headings)} heading chunks, sections={sections}")


# ---------- 5. 噪声过滤 ----------
SAMPLE_NOISY = {
    "title": "夏日穿搭分享",
    "body": """今天分享 5 套夏日穿搭。

第一套是米色亚麻衬衫 + 阔腿裤，凉快又高级。

————

未完待续

————

第二套是白 T + 牛仔半裙。
""",
}


def test_noise_filtered():
    """'————'、'未完待续' 等无意义段落应被丢弃。"""
    chunks = _split_with_title(SAMPLE_NOISY["body"], title=SAMPLE_NOISY["title"])
    texts = [c.text for c in chunks]
    # 显式断言"未完待续"和"————"都不在
    for t in texts:
        assert "未完待续" not in t, f"noise '未完待续' leaked: {t}"
        assert "————" not in t, f"noise '————' leaked: {t}"
    print(f"  ✓ test_noise_filtered: {len(chunks)} chunks, all noise removed")


# ---------- 6. 边界条件 ----------
def test_empty_text():
    """空 body + 只有标题的笔记。"""
    chunks = _split_with_title("", title="只有标题的笔记")
    assert len(chunks) == 1
    assert chunks[0].type == "heading"
    assert chunks[0].section == "标题"
    print(f"  ✓ test_empty_text: returns only the title chunk")


def test_no_title():
    """没有 title 的纯文本。"""
    chunks = _split_with_title("第一段内容。\n\n第二段内容。", title="")
    assert len(chunks) == 2
    assert all(c.section == "正文" for c in chunks)
    assert chunks[0].char_start == 0
    print(f"  ✓ test_no_title: 2 body chunks, offsets start at 0")


def test_chunk_index_continuous():
    """所有 chunk 的 index 字段应当从 0 连续递增。"""
    chunks = _split_with_title(SAMPLE_LONG["body"], title=SAMPLE_LONG["title"])
    for i, c in enumerate(chunks):
        assert c.index == i, f"index not continuous at {i}: got {c.index}"
    print(f"  ✓ test_chunk_index_continuous: {len(chunks)} chunks, indices 0..{len(chunks)-1}")


def test_to_dict_serializable():
    """to_dict 输出应该是可 JSON 序列化的（落库前提）。"""
    import json
    chunks = _split_with_title(SAMPLE_SHORT["body"], title=SAMPLE_SHORT["title"])
    payload = [c.to_dict() for c in chunks]
    s = json.dumps(payload, ensure_ascii=False)
    assert "上海" in s
    print(f"  ✓ test_to_dict_serializable: {len(payload)} chunks, JSON ok")


# ---------- 跑测试 ----------
def run_all():
    """不用 pytest，直接 python 跑。"""
    tests = [
        test_short_note_basic,
        test_short_note_offset_monotonic,
        test_short_note_offset_can_locate_title,
        test_long_paragraph_triggers_window,
        test_long_paragraph_window_size,
        test_list_paragraph_recognized,
        test_headings_and_sections,
        test_noise_filtered,
        test_empty_text,
        test_no_title,
        test_chunk_index_continuous,
        test_to_dict_serializable,
    ]
    print(f"Running {len(tests)} tests...\n")
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"  ✗ {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ {t.__name__}: {type(e).__name__}: {e}")
            failed += 1

    print()
    if failed == 0:
        print(f"✅ All {len(tests)} tests passed.")
    else:
        print(f"❌ {failed}/{len(tests)} tests failed.")
        sys.exit(1)


if __name__ == "__main__":
    run_all()
