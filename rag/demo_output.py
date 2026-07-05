"""
样例运行：把 chunker.py 的所有 demo 跑一遍，输出可视化结果。
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from chunker import split, insert_with_title


SAMPLES = [
    {
        "name": "SAMPLE 1: 短笔记（小红书探店）",
        "title": "上海宝藏日料店｜藏在愚园路的深夜食堂，3 样必点",
        "body": """路过愚园路偶然发现的小店，门面非常不起眼，但推门进去别有洞天。

3 样必点

① 味噌烤银鳕鱼：外皮微焦内里嫩滑，咸甜平衡。
② 手作玉子烧：现点现做等 15 分钟，但是值得。
③ 柚子盐拉面：汤底清爽不腻，柚子香气很正。

避雷明太鱼籽茶泡饭，太咸。""",
    },
    {
        "name": "SAMPLE 2: 长段落（RAG 笔记）",
        "title": "RAG 实战笔记：Chunk 切分的 5 种策略对比",
        "body": """在 RAG 系统里，Chunk 切分是整个 pipeline 中最容易被忽视、但影响最大的一环。切得太粗，检索粒度太粗、答案不精准；切得太细，又会丢上下文；切得不对齐语义边界，LLM 读起来会感到割裂；切得太规则，又会让某些内容长度失控。综合来看，切分策略的选择要兼顾语义完整、检索粒度、实现成本、调试难度四个维度。一种好的切分策略应该满足：chunk 内信息密度均匀，chunk 之间不重复太多，溯源时定位精确，导入时不要慢到用户受不了。

第一种是固定 token 切分，按 256 或 512 token 强行切一刀。实现最简单，但经常会切断语义，比如把一个完整的论点一分为二，让 LLM 读起来很费劲。在英文场景下还好，中文里 token 边界和字边界不一致，问题更明显。我们团队曾经在 1 万篇小红书笔记上做过实验，固定 256 token 切分的召回准确率只有 62%，明显低于段落切分。""",
    },
    {
        "name": "SAMPLE 3: 列表笔记（护肤）",
        "title": "敏感肌夏季护肤｜成分党整理的 6 款无雷单品",
        "body": """夏天敏感肌容易泛红闷痘，下面是 6 款我反复回购的单品。

- 洁面：珂润润浸保湿洁颜泡沫。氨基酸体系，挤出来就是泡沫，懒人友好。
- 精华：薇诺娜舒敏保湿特护精华。马齿苋提取物舒缓泛红，质地清爽。
- 面霜：理肤泉 B5 修复面霜。b5 + 积雪草，烂脸期救星，略油。
- 防晒：怡丽丝尔小金管。物化结合，成膜快不搓泥。
- 卸妆：Fancl 速净卸妆油。乳化快，不闷痘，无香料。
- 面膜：可复美重组胶原蛋白面膜。医美术后修复用，敏感肌一周 1-2 次即可。

以上 6 款覆盖洁面到面膜，可以闭眼入。""",
    },
    {
        "name": "SAMPLE 4: 标题 + 章节（京都旅行）",
        "title": "京都 5 月自由行攻略",
        "body": """# 5 月为什么适合去京都

5 月是京都最舒服的季节，气温 18-25°C，既不冷也不热。新绿和樱花尾季叠加，人比 4 月少一半。

# 行程推荐

## Day 1：岚山 + 渡月桥

清晨 6 点去岚山竹林，游客还没进来，随便拍都是大片。渡月桥边上有家豆腐料理店必去。

## Day 2：银阁寺 + 哲学之道

银阁寺清晨 7 点开门，哲学之道沿途 3 家町家咖啡，最推荐 % Arabica Kyoto。""",
    },
]


def hr(label):
    print()
    print("=" * 70)
    print(f"  {label}")
    print("=" * 70)


def show_sample(sample):
    hr(sample["name"])
    chunks = insert_with_title(sample["title"], split(sample["body"]))
    print(f"  → 共 {len(chunks)} 个 chunks\n")
    for c in chunks:
        # type badge
        type_emoji = {
            "heading": "🏷️",
            "paragraph": "📝",
            "list": "📋",
            "windowed": "✂️",
        }.get(c.type, "•")
        print(f"  [{c.index:02d}] {type_emoji} {c.type:9s} | {c.section:4s} "
              f"| [{c.char_start:3d}:{c.char_end:3d}]")
        text_preview = c.text if len(c.text) <= 60 else c.text[:57] + "..."
        print(f"       └ {text_preview}")
    print()


def main():
    print("\n" + "▓" * 70)
    print("  KnowFlow Chunker · Demo Output")
    print("▓" * 70)
    for s in SAMPLES:
        show_sample(s)
    print("=" * 70)
    print("  ✅ Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()
