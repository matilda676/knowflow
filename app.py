from __future__ import annotations

import base64
import hmac
import json
import os
import re
import sqlite3
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from flask import Flask, jsonify, request, send_from_directory
from werkzeug.exceptions import HTTPException

from rag.chunker import insert_with_title, split
from rag.llm import generate_answer as llm_generate_answer
from rag.llm import generate_note_summary as llm_generate_note_summary
from rag.llm import llm_info

try:
    from rag.embedding import embedding_info
    from rag.vector_store import query_chunks as vector_query_chunks
    from rag.vector_store import reset_collection, upsert_chunks
    VECTOR_AVAILABLE = True
except Exception:
    VECTOR_AVAILABLE = False
    embedding_info = None
    vector_query_chunks = None
    reset_collection = None
    upsert_chunks = None


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("KNOWFLOW_DATA_DIR", str(BASE_DIR / "data"))).expanduser()
DB_PATH = DATA_DIR / "knowflow.db"


app = Flask(__name__, static_folder=None)


def access_password() -> str:
    return os.getenv("KNOWFLOW_ACCESS_PASSWORD", "").strip()


def check_basic_auth() -> bool:
    password = access_password()
    if not password:
        return True

    auth_header = request.headers.get("Authorization", "")
    scheme, _, token = auth_header.partition(" ")
    if scheme.lower() != "basic" or not token:
        return False

    try:
        decoded = base64.b64decode(token).decode("utf-8")
    except Exception:
        return False

    username, _, supplied_password = decoded.partition(":")
    return hmac.compare_digest(username, "knowflow") and hmac.compare_digest(supplied_password, password)


def auth_required_response():
    response = jsonify({"error": "需要访问密码"})
    response.status_code = 401
    response.headers["WWW-Authenticate"] = 'Basic realm="KnowFlow"'
    return response


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_db() -> sqlite3.Connection:
    DATA_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_db() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                body TEXT NOT NULL,
                category TEXT NOT NULL DEFAULT '未分类',
                summary TEXT NOT NULL DEFAULT '',
                fields_json TEXT NOT NULL DEFAULT '{}',
                source_url TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                note_id INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                type TEXT NOT NULL,
                section TEXT NOT NULL,
                char_start INTEGER NOT NULL,
                char_end INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (note_id) REFERENCES notes(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_chunks_note_id ON chunks(note_id);
            CREATE INDEX IF NOT EXISTS idx_notes_category ON notes(category);
            """
        )


def row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return dict(row)


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def make_summary(body: str, max_len: int = 90) -> str:
    text = normalize_space(body)
    if len(text) <= max_len:
        return text
    return text[:max_len].rstrip() + "..."


def split_sentences(text: str) -> list[str]:
    return [
        normalize_space(part)
        for part in re.split(r"[。！？!?\n]+", text)
        if len(normalize_space(part)) >= 8
    ]


def extract_numbered_lines(text: str) -> list[str]:
    lines = []
    for line in text.splitlines():
        cleaned = normalize_space(line)
        if re.match(r"^(?:\d+[.\)、]|[①②③④⑤⑥⑦⑧⑨⑩]|[1-9]️⃣)", cleaned):
            lines.append(re.sub(r"^(?:\d+[.\)、]|[①②③④⑤⑥⑦⑧⑨⑩]|[1-9]️⃣)\s*", "", cleaned))
    return lines


def make_note_summary(title: str, body: str, category: str, max_len: int = 170) -> str:
    text = normalize_space(body)
    if not text:
        return "暂无摘要"

    sentences = split_sentences(body)
    if category == "旅行":
        places = [place for place in ["大阪", "京都", "东京", "富士山", "奈良", "神户", "关西", "关东"] if place in f"{title}{body}"]
        days = re.search(r"(\d+\s*[天日]\s*\d*\s*[晚夜]?)", f"{title} {body}")
        highlights = [
            sentence
            for sentence in sentences
            if any(word in sentence for word in ["推荐", "交通", "酒店", "路线", "神社", "机场", "漂亮", "体验"])
        ][:2]
        parts = []
        if days or places:
            parts.append(f"这是一篇{days.group(1).replace(' ', '') if days else ''}{'、'.join(places[:4])}自由行笔记，重点是路线和交通安排。")
        if highlights:
            parts.append("可参考：" + "；".join(highlights))
        summary = "".join(parts) or make_summary(body, max_len)
    elif category == "美食":
        recs = extract_recommendations(body)
        price = extract_price(body)
        avoids = extract_avoid_items(body)
        parts = []
        if recs:
            parts.append(f"推荐关注：{'、'.join(recs[:3])}。")
        if price:
            parts.append(f"人均约 {price}。")
        if avoids:
            parts.append(f"避雷：{'、'.join(avoids)}。")
        summary = "".join(parts) or make_summary(body, max_len)
    elif category == "护肤":
        items = extract_numbered_lines(body)
        parts = []
        if items:
            item_text = "；".join(items[:3])
            parts.append(f"这篇主要记录 {len(items)} 个护肤步骤/单品：{item_text}。")
        effect_keywords = []
        if "滋润" in body or "巨润" in body:
            effect_keywords.append("更滋润")
        if "纹路" in body:
            effect_keywords.append("浅纹变淡")
        if "下颚线" in body:
            effect_keywords.append("下颚线更清晰")
        if "卡粉" in body or "卡纹" in body:
            effect_keywords.append("上妆不卡粉卡纹")
        if "透亮" in body:
            effect_keywords.append("肤感更透亮")
        if effect_keywords:
            parts.append(f"作者主观反馈：{ '、'.join(effect_keywords[:5]) }。")
        if items and any(word in body for word in ["精华水", "精华油", "刮痧"]):
            parts.append("适合当作护肤流程参考，不等同于成分功效验证。")
        summary = "".join(parts) or make_summary(body, max_len)
    else:
        summary = "；".join(sentences[:2]) or make_summary(body, max_len)

    if len(summary) <= max_len:
        return summary
    return summary[:max_len].rstrip() + "..."


def infer_category(title: str, body: str) -> str:
    text = f"{title} {body}"
    rules = [
        ("美食", ["日料", "餐厅", "必点", "人均", "拉面", "咖啡", "探店", "菜"]),
        ("旅行", ["攻略", "行程", "酒店", "机票", "景点", "京都", "上海", "路线"]),
        ("护肤", ["护肤", "敏感肌", "防晒", "面霜", "精华", "洁面", "成分"]),
        ("学习", ["学习", "笔记", "课程", "方法", "复盘", "效率", "考试"]),
        ("家居", ["家居", "收纳", "装修", "厨房", "卧室", "清洁"]),
        ("产品 · 增长", ["产品", "增长", "用户", "转化", "留存", "MVP", "PRD"]),
    ]
    for category, keywords in rules:
        if any(keyword.lower() in text.lower() for keyword in keywords):
            return category
    return "未分类"


def extract_fields(title: str, body: str, category: str) -> dict[str, str]:
    text = f"{title}\n{body}"
    fields: dict[str, str] = {
        "主题": title,
        "分类": category,
    }

    price = re.search(r"(人均|预算|价格|花费)[：:\s]*([0-9]+(?:\.[0-9]+)?\s*(?:元|块|rmb|RMB)?)", text)
    if price:
        fields["价格"] = price.group(2)

    location = re.search(r"((?:上海|北京|广州|深圳|杭州|成都|京都|东京|大阪)[^\n，。,.、]{0,12})", text)
    if location:
        fields["地点"] = location.group(1)

    must_try = re.search(r"(必点|推荐|值得)[：:\s]*([^\n。]+)", text)
    if must_try:
        fields["推荐项"] = must_try.group(2).strip()

    avoid = re.search(r"(避雷|不推荐|踩雷)[：:\s]*([^\n。]+)", text)
    if avoid:
        fields["避雷项"] = avoid.group(2).strip()

    if category == "美食":
        shop = re.search(r"(店名)[=：:\s]*([^\n/，。]+)", text)
        if shop:
            fields["店名"] = shop.group(2).strip()
        if "推荐项" in fields:
            fields["必点菜"] = fields["推荐项"]

    return fields


def create_note(payload: dict[str, Any]) -> dict[str, Any]:
    title = normalize_space(str(payload.get("title") or "未命名笔记"))
    body = str(payload.get("body") or "").strip()
    if not body:
        raise ValueError("body is required")

    category = normalize_space(str(payload.get("category") or "")) or infer_category(title, body)
    summary = (
        normalize_space(str(payload.get("summary") or ""))
        or llm_generate_note_summary(title, body, category)
        or make_note_summary(title, body, category)
    )
    source_url = normalize_space(str(payload.get("source_url") or ""))
    fields = payload.get("fields")
    if not isinstance(fields, dict) or not fields:
        fields = extract_fields(title, body, category)

    created_at = now_iso()
    body_chunks = split(body)
    chunks = insert_with_title(title, body_chunks)

    vector_payload: list[dict[str, Any]] = []
    with get_db() as conn:
        cur = conn.execute(
            """
            INSERT INTO notes (title, body, category, summary, fields_json, source_url, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (title, body, category, summary, json.dumps(fields, ensure_ascii=False), source_url, created_at),
        )
        note_id = int(cur.lastrowid)
        conn.executemany(
            """
            INSERT INTO chunks
            (note_id, chunk_index, text, type, section, char_start, char_end, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    note_id,
                    c.index,
                    c.text,
                    c.type,
                    c.section,
                    c.char_start,
                    c.char_end,
                    created_at,
                )
                for c in chunks
            ],
        )
        vector_payload = [
            {
                "note_id": note_id,
                "chunk_index": c.index,
                "text": c.text,
                "type": c.type,
                "section": c.section,
                "char_start": c.char_start,
                "char_end": c.char_end,
                "title": title,
                "category": category,
            }
            for c in chunks
        ]

    if VECTOR_AVAILABLE and upsert_chunks is not None:
        upsert_chunks(vector_payload)

    return get_note(note_id) | {"chunk_count": len(chunks)}


def get_note(note_id: int) -> dict[str, Any]:
    with get_db() as conn:
        note = conn.execute("SELECT * FROM notes WHERE id = ?", (note_id,)).fetchone()
        if note is None:
            raise KeyError("note not found")
        chunks = conn.execute(
            "SELECT chunk_index, text, type, section, char_start, char_end FROM chunks WHERE note_id = ? ORDER BY chunk_index",
            (note_id,),
        ).fetchall()

    data = row_to_dict(note)
    data["fields"] = json.loads(data.pop("fields_json") or "{}")
    data["chunks"] = [row_to_dict(chunk) for chunk in chunks]
    data["summary"] = make_note_summary(str(data["title"]), str(data["body"]), str(data["category"]))
    return data


def delete_note(note_id: int) -> bool:
    with get_db() as conn:
        note = conn.execute("SELECT id FROM notes WHERE id = ?", (note_id,)).fetchone()
        if note is None:
            return False
        conn.execute("DELETE FROM chunks WHERE note_id = ?", (note_id,))
        conn.execute("DELETE FROM notes WHERE id = ?", (note_id,))

    if VECTOR_AVAILABLE and reset_collection is not None and upsert_chunks is not None:
        rebuild_vector_index()
    return True


def tokenize(text: str) -> list[str]:
    lowered = text.lower()
    words = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]{2,}", lowered)
    chars = [char for char in lowered if "\u4e00" <= char <= "\u9fff"]
    return words + chars


TRAVEL_LOCATION_GROUPS = [
    ["泰国", "曼谷", "普吉", "普吉岛", "芭提雅", "清迈", "清莱", "thailand", "bangkok", "phuket", "pattaya"],
    ["日本", "关西", "关东", "大阪", "京都", "东京", "富士山", "奈良", "神户", "japan", "osaka", "kyoto", "tokyo"],
    ["中国", "上海", "北京", "广州", "深圳", "杭州", "成都", "重庆"],
]


def detect_location_terms(query: str) -> list[str]:
    query_text = query.lower()
    for group in TRAVEL_LOCATION_GROUPS:
        if any(term.lower() in query_text for term in group):
            return group
    return []


def detect_exact_location_terms(query: str) -> list[str]:
    query_text = query.lower()
    terms: list[str] = []
    for group in TRAVEL_LOCATION_GROUPS:
        for term in group:
            if term.lower() in query_text:
                terms.append(term)
    return terms


def match_location_score(item: dict[str, Any], location_terms: list[str]) -> int:
    if not location_terms:
        return 0
    haystack = f"{item.get('title') or ''} {item.get('text') or ''}".lower()
    return sum(1 for term in location_terms if term.lower() in haystack)


def apply_location_filter(matches: list[dict[str, Any]], query: str) -> list[dict[str, Any]]:
    location_terms = detect_location_terms(query)
    if not location_terms:
        return matches

    exact_terms = detect_exact_location_terms(query)
    filtered = []
    for item in matches:
        broad_score = match_location_score(item, location_terms)
        exact_score = match_location_score(item, exact_terms)
        location_score = broad_score + exact_score * 5
        if location_score <= 0:
            continue
        item = dict(item)
        item["location_score"] = location_score
        filtered.append(item)
    filtered.sort(key=lambda item: (-int(item.get("location_score") or 0), -float(item.get("score") or 0)))
    return filtered


def search_chunks(query: str, *, category: Optional[str] = None, limit: int = 5) -> list[dict[str, Any]]:
    if VECTOR_AVAILABLE and vector_query_chunks is not None:
        try:
            vector_matches = vector_query_chunks(query, category=category, limit=max(limit * 4, limit))
            vector_matches = apply_location_filter(vector_matches, query)
            if vector_matches:
                return vector_matches[:limit]
        except Exception:
            pass

    tokens = tokenize(query)
    if not tokens:
        return []

    sql = """
        SELECT
            chunks.id AS chunk_id,
            chunks.note_id,
            chunks.chunk_index,
            chunks.text,
            chunks.type,
            chunks.section,
            chunks.char_start,
            chunks.char_end,
            notes.title,
            notes.category
        FROM chunks
        JOIN notes ON notes.id = chunks.note_id
    """
    params: list[Any] = []
    if category:
        sql += " WHERE notes.category = ?"
        params.append(category)
    sql += " ORDER BY notes.created_at DESC, chunks.chunk_index ASC"

    scored: list[dict[str, Any]] = []
    with get_db() as conn:
        for row in conn.execute(sql, params).fetchall():
            item = row_to_dict(row)
            haystack = f"{item['title']} {item['category']} {item['text']}".lower()
            score = sum(3 if token in item["title"].lower() else 1 for token in tokens if token in haystack)
            if score > 0:
                item["score"] = score
                scored.append(item)

    scored.sort(key=lambda item: (-item["score"], item["note_id"], item["chunk_index"]))
    scored = apply_location_filter(scored, query) or ([] if detect_location_terms(query) else scored)
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for item in scored:
        key = (item["note_id"], normalize_space(item["text"]))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
        if len(deduped) >= limit:
            break
    return deduped


def extract_recommendations(text: str) -> list[str]:
    candidates: list[str] = []

    patterns = [
        r"(?:必点|推荐|值得点|可以点)[：:\s]*([^。！？!?\n]+)",
        r"[①②③④⑤⑥⑦⑧⑨⑩]\s*([^：:，,。！？!?\n]+)",
        r"\d+[.\)、]\s*([^：:，,。！？!?\n]+)",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            segment = match.group(1).strip()
            parts = re.split(r"[、，,/\s]+", segment)
            for part in parts:
                item = part.strip("：:；;。.!！?？ ")
                if 2 <= len(item) <= 18 and not any(stop in item for stop in ["避雷", "不推荐", "人均", "门面", "小店", "必点"]):
                    candidates.append(item)

    if not candidates:
        food_words = re.findall(r"[\u4e00-\u9fff]{2,12}(?:鱼|烧|拉面|茶泡饭|咖啡|面|饭|寿司|刺身|甜品|蛋糕)", text)
        candidates.extend(food_words)

    unique: list[str] = []
    for item in candidates:
        if item not in unique:
            unique.append(item)
    return unique[:5]


def is_detail_query(query: str) -> bool:
    return any(word in query.lower() for word in [
        "必点", "推荐", "吃什么", "点什么", "菜", "避雷", "别点", "不推荐", "踩雷", "人均", "价格", "多少钱", "预算",
    ])


def is_travel_query(query: str) -> bool:
    query_text = query.lower()
    return any(word in query_text for word in [
        "旅行", "旅游", "行程", "四天", "三晚", "4天", "3晚", "一个人", "solo", "自由行", "日本", "关西", "关东", "京都", "大阪", "东京",
        "泰国", "曼谷", "普吉", "芭提雅", "清迈",
    ])


def infer_query_category(query: str) -> Optional[str]:
    if is_travel_query(query):
        return "旅行"
    query_text = query.lower()
    if any(word in query_text for word in ["吃", "餐厅", "日料", "人均", "必点", "菜", "咖啡", "烤肉", "拉面"]):
        return "美食"
    return None


def note_chunks_for_answer(note_id: int) -> list[dict[str, Any]]:
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT
                chunks.id AS chunk_id,
                chunks.note_id,
                chunks.chunk_index,
                chunks.text,
                chunks.type,
                chunks.section,
                chunks.char_start,
                chunks.char_end,
                notes.title,
                notes.category
            FROM chunks
            JOIN notes ON notes.id = chunks.note_id
            WHERE notes.id = ?
            ORDER BY
                CASE
                    WHEN chunks.type = 'heading' THEN 4
                    WHEN chunks.text LIKE '%①%' OR chunks.text LIKE '%推荐%' THEN 0
                    WHEN chunks.text LIKE '%必点%' THEN 1
                    WHEN chunks.text LIKE '%避雷%' OR chunks.text LIKE '%人均%' THEN 1
                    ELSE 2
                END,
                chunks.chunk_index ASC
            """,
            (note_id,),
        ).fetchall()
    return [row_to_dict(row) for row in rows]


def rebuild_vector_index() -> int:
    if not VECTOR_AVAILABLE or reset_collection is None or upsert_chunks is None:
        return 0

    reset_collection()
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT
                chunks.note_id,
                chunks.chunk_index,
                chunks.text,
                chunks.type,
                chunks.section,
                chunks.char_start,
                chunks.char_end,
                notes.title,
                notes.category
            FROM chunks
            JOIN notes ON notes.id = chunks.note_id
            ORDER BY chunks.note_id, chunks.chunk_index
            """
        ).fetchall()
    payload = [row_to_dict(row) for row in rows]
    upsert_chunks(payload)
    return len(payload)


def extract_avoid_items(text: str) -> list[str]:
    items: list[str] = []
    for match in re.finditer(r"(?:避雷|不推荐|踩雷)[：:\s]*([^。！？!?\n]+)", text):
        for part in re.split(r"[、，,/\s]+", match.group(1)):
            item = part.strip("：:；;。.!！?？ ")
            if 2 <= len(item) <= 18 and item not in {"太咸", "太甜", "太油", "一般"}:
                items.append(item)
    unique: list[str] = []
    for item in items:
        if item not in unique:
            unique.append(item)
    return unique[:3]


def extract_price(text: str) -> Optional[str]:
    match = re.search(r"(人均|预算|价格|花费)[：:\s]*([0-9]+(?:\.[0-9]+)?\s*(?:元|块|rmb|RMB)?)", text)
    if match:
        return match.group(2)
    match = re.search(r"(人均|预算|价格|花费)[^。！？!?\n]{0,12}?((?:[一二三四五六七八九十百两]+|[0-9]+)[多几]?(?:十|百)?(?:来)?(?:元|块)?|五六十|六七十|四五十)", text)
    if match:
        price = match.group(2)
        return price if price.endswith(("元", "块")) else f"{price}元左右"
    return None


def compact_matches_by_note(matches: list[dict[str, Any]], *, limit: int = 5) -> list[dict[str, Any]]:
    compacted: list[dict[str, Any]] = []
    seen: set[int] = set()
    for match in matches:
        note_id = int(match.get("note_id") or 0)
        if note_id in seen:
            continue
        seen.add(note_id)
        compacted.append(match)
        if len(compacted) >= limit:
            break
    return compacted


def expand_top_notes(matches: list[dict[str, Any]], *, max_notes: int = 3, max_chunks_per_note: int = 4) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for match in compact_matches_by_note(matches, limit=max_notes):
        chunks = note_chunks_for_answer(int(match["note_id"])) or [match]
        expanded.extend(chunks[:max_chunks_per_note])
    return expanded or matches


def representative_sentences(text: str, query: str, limit: int = 5, skip_phrases: Optional[list[str]] = None) -> list[str]:
    query_tokens = tokenize(query)
    skip_phrases = skip_phrases or []
    sentences = [
        normalize_space(part)
        for part in re.split(r"[。！？!?\n]+", text)
        if len(normalize_space(part)) >= 8
    ]

    def score(sentence: str) -> tuple[int, int]:
        lowered = sentence.lower()
        token_score = sum(1 for token in query_tokens if token and token in lowered)
        detail_score = sum(1 for word in ["推荐", "适合", "路线", "行程", "人均", "预算", "避雷", "必点", "清晨", "晚上"] if word in sentence)
        return (token_score + detail_score, -len(sentence))

    ranked = sorted(sentences, key=score, reverse=True)
    picked: list[str] = []
    for sentence in ranked:
        if any(sentence == phrase or sentence in phrase for phrase in skip_phrases):
            continue
        if sentence in picked:
            continue
        picked.append(sentence)
        if len(picked) >= limit:
            break
    return picked


def make_answer(query: str, matches: list[dict[str, Any]]) -> str:
    if not matches:
        return "当前知识库里没有找到相关内容。你可以先导入相关笔记，再继续提问。"

    query_text = query.lower()
    note_titles = []
    for match in matches:
        title = str(match.get("title") or "")
        if title and title not in note_titles:
            note_titles.append(title)
    primary_note_id = matches[0].get("note_id") if matches else None
    primary_texts = [
        match["text"]
        for match in matches
        if match.get("note_id") == primary_note_id
        and normalize_space(str(match.get("text") or "")) not in note_titles
    ]
    primary_combined = "\n".join(primary_texts)
    combined = "\n".join(
        match["text"]
        for match in matches
        if normalize_space(str(match.get("text") or "")) not in note_titles
    )
    if not combined.strip():
        combined = "\n".join(match["text"] for match in matches)
    recs = extract_recommendations(combined)
    avoids = extract_avoid_items(combined)
    price = extract_price(combined)
    source_line = f"我主要参考了《{note_titles[0]}》" if note_titles else "我主要参考了你导入的笔记"

    if is_travel_query(query):
        points = representative_sentences(combined, query, limit=5, skip_phrases=note_titles)
        has_short_trip = any(word in query_text for word in ["四天", "三晚", "4天", "3晚"])
        location_terms = detect_location_terms(query)
        is_japan_query = bool(location_terms) and "日本" in location_terms
        is_thailand_query = bool(location_terms) and "泰国" in location_terms
        lines = [f"{source_line}。"]
        if points:
            lines.append("从笔记里能直接用上的信息是：")
            lines.extend(f"{idx}. {point}" for idx, point in enumerate(points[:3], start=1))
        if is_thailand_query:
            lines.append("按当前库里的内容看，泰国路线可以优先在“曼谷 + 普吉”或“曼谷 + 芭提雅”之间选一条，不建议把所有城市都塞进同一趟短行程。")
            if has_short_trip:
                lines.append("如果时间短，更稳的是只保留曼谷加一个海岛/近郊目的地，减少换城市成本。")
        elif is_japan_query:
            if has_short_trip:
                lines.append("如果是四天三晚，更稳的选择是只截取其中一段：优先大阪 + 京都；如果你更想看城市感，再选东京 + 近郊。")
            else:
                lines.append("按库里的日本笔记看，可以优先围绕大阪、京都、东京或富士山这些节点组合路线。")
        else:
            lines.append("建议先按同一目的地里的高频路线做取舍，减少跨城市移动，把时间留给真正想体验的景点和餐厅。")
        lines.append("下方来源只按笔记展示，点进去可以看完整原文。")
        return "\n".join(lines)

    if any(word in query_text for word in ["避雷", "别点", "不推荐", "踩雷"]):
        if avoids:
            return "\n".join([
                f"{source_line}。不建议优先考虑：{'、'.join(avoids)}。",
                "如果你要做选择，可以先把这些项排除，再看下方来源回到原笔记核对细节。",
            ])

    if any(word in query_text for word in ["必点", "推荐", "吃什么", "点什么", "菜"]):
        if recs:
            lines = [f"推荐点：{'、'.join(recs[:3])}。"]
            if price:
                lines.append(f"这篇笔记里提到的人均大约是 {price}。")
            if avoids:
                lines.append(f"避雷：{'、'.join(avoids)}。")
            lines.append("我把相关片段合并后给你这个结论，下方只保留对应原笔记入口。")
            return "\n".join(lines)

    if any(word in query_text for word in ["人均", "价格", "多少钱", "预算"]):
        if price:
            return "\n".join([
                f"{source_line}。笔记里提到的人均大约是 {price}。",
                "如果你要控制预算，可以把这条作为预估，再点下方来源回到原笔记看搭配细节。",
            ])

    if any(word in query_text for word in ["餐厅", "一人食", "一个人吃", "独自吃饭", "吃饭"]):
        evidence = primary_combined or combined
        evidence_price = extract_price(evidence)
        points = representative_sentences(
            evidence,
            query,
            limit=3,
            skip_phrases=note_titles,
        )
        lines = [f"{source_line}，这家更适合当作一个人吃饭的备选。"]
        if evidence_price:
            lines.append(f"价格：笔记里提到人均大约 {evidence_price}。")
        if points:
            lines.append("体验重点：")
            lines.extend(f"{idx}. {point}" for idx, point in enumerate(points, start=1))
        lines.append("建议你点进下方来源看完整原文，再决定是不是符合你的口味和预算。")
        return "\n".join(lines)

    points = representative_sentences(combined, query, limit=4, skip_phrases=note_titles)
    lines = [f"{source_line}，我给你的直接结论是："]
    if points:
        lines.extend(f"{idx}. {point}" for idx, point in enumerate(points, start=1))
    else:
        lines.append(normalize_space(matches[0]["text"])[:160])
    lines.append("我已经把同一篇笔记的多个片段合并理解了，下方来源按原笔记去重展示。")
    return "\n".join(lines)


@app.before_request
def require_access_password():
    if request.method == "OPTIONS":
        return None
    if check_basic_auth():
        return None
    return auth_required_response()


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,DELETE,OPTIONS"
    return response


@app.errorhandler(Exception)
def handle_api_error(error):
    if not request.path.startswith("/api/"):
        raise error

    if isinstance(error, HTTPException):
        return jsonify({"error": error.description or error.name}), error.code

    traceback.print_exc()
    return jsonify({"error": "后端处理失败，已自动保留错误日志。请稍后重试。"}), 500


@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "chat.html")


@app.route("/<path:filename>")
def static_files(filename: str):
    allowed = {"index.html", "chat.html", "import.html", "library.html", "detail.html", "tech_glossary.html", "knowflow-static.js"}
    if filename in allowed:
        return send_from_directory(BASE_DIR, filename)
    return jsonify({"error": "not found"}), 404


@app.route("/api/health")
def health():
    embedding = embedding_info() if VECTOR_AVAILABLE and embedding_info is not None else None
    return jsonify({
        "ok": True,
        "service": "KnowFlow API",
        "db": str(DB_PATH),
        "vector_available": VECTOR_AVAILABLE,
        "embedding": embedding,
        "llm": llm_info(),
    })


@app.route("/api/notes", methods=["GET", "POST", "OPTIONS"])
def notes():
    if request.method == "OPTIONS":
        return ("", 204)

    if request.method == "POST":
        try:
            note = create_note(request.get_json(force=True) or {})
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify({"note": note}), 201

    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT
                notes.id,
                notes.title,
                notes.category,
                notes.summary,
                notes.source_url,
                notes.created_at,
                COUNT(chunks.id) AS chunk_count
            FROM notes
            LEFT JOIN chunks ON chunks.note_id = notes.id
            GROUP BY notes.id
            ORDER BY notes.created_at DESC
            """
        ).fetchall()
    return jsonify({"notes": [row_to_dict(row) for row in rows]})


@app.route("/api/notes/<int:note_id>")
def note_detail(note_id: int):
    try:
        note = get_note(note_id)
    except KeyError:
        return jsonify({"error": "note not found"}), 404
    return jsonify({"note": note})


@app.route("/api/notes/<int:note_id>", methods=["DELETE"])
def remove_note(note_id: int):
    if not delete_note(note_id):
        return jsonify({"error": "note not found"}), 404
    return jsonify({"ok": True, "deleted_note_id": note_id})


@app.route("/api/search")
def search():
    query = request.args.get("q", "").strip()
    category = request.args.get("category") or None
    matches = search_chunks(query, category=category, limit=10)
    return jsonify({"query": query, "matches": matches})


@app.route("/api/reindex", methods=["POST"])
def reindex():
    if not VECTOR_AVAILABLE:
        return jsonify({"error": "ChromaDB is not installed"}), 503
    count = rebuild_vector_index()
    embedding = embedding_info() if embedding_info is not None else None
    return jsonify({"ok": True, "indexed_chunks": count, "embedding": embedding})


@app.route("/api/chat", methods=["POST", "OPTIONS"])
def chat():
    if request.method == "OPTIONS":
        return ("", 204)

    payload = request.get_json(force=True) or {}
    query = str(payload.get("message") or payload.get("query") or "").strip()
    category = str(payload.get("category") or "").strip() or infer_query_category(query)
    if not query:
        return jsonify({"error": "message is required"}), 400

    matches = search_chunks(query, category=category, limit=12)
    answer_matches = expand_top_notes(matches, max_notes=3, max_chunks_per_note=5)
    citation_matches = compact_matches_by_note(matches, limit=5)

    citations = [
        {
            "number": idx,
            "note_id": match["note_id"],
            "title": match["title"],
            "category": match["category"],
            "chunk_index": match["chunk_index"],
            "char_start": match["char_start"],
            "char_end": match["char_end"],
            "text": match["text"],
        }
        for idx, match in enumerate(citation_matches, start=1)
    ]
    llm_answer = llm_generate_answer(query, answer_matches)
    answer = llm_answer or make_answer(query, answer_matches)
    llm = llm_info()
    return jsonify(
        {
            "answer": answer,
            "citations": citations,
            "debug": {
                "retrieved": len(answer_matches),
                "mode": "llm_rag" if llm_answer else ("vector_chroma" if VECTOR_AVAILABLE else "keyword_mvp"),
                "embedding": embedding_info() if VECTOR_AVAILABLE and embedding_info is not None else None,
                "llm": llm,
            },
        }
    )


init_db()
if VECTOR_AVAILABLE:
    try:
        rebuild_vector_index()
    except Exception:
        pass


if __name__ == "__main__":
    host = os.getenv("KNOWFLOW_HOST", "0.0.0.0")
    port = int(os.getenv("KNOWFLOW_PORT", "5001"))
    app.run(host=host, port=port, debug=False, use_reloader=False)
