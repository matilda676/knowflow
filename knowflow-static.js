(function () {
  const STORAGE_KEY = "knowflow_notes_v1";

  function nowISO() {
    return new Date().toISOString();
  }

  function normalizeSpace(text) {
    return String(text || "").replace(/\s+/g, " ").trim();
  }

  function splitText(text) {
    const normalized = String(text || "").trim();
    if (!normalized) return [];
    const parts = normalized
      .split(/\n\s*\n|(?<=[。！？!?])\s+/)
      .map((part) => normalizeSpace(part))
      .filter(Boolean);
    const chunks = [];
    let index = 0;
    parts.forEach((part) => {
      if (part.length <= 260) {
        chunks.push(part);
        return;
      }
      for (let start = 0; start < part.length; start += 220) {
        chunks.push(part.slice(start, start + 260));
      }
    });
    return chunks.map((text) => ({
      chunk_index: index++,
      text,
      type: "windowed",
      section: "",
      char_start: 0,
      char_end: text.length,
    }));
  }

  function inferCategory(title, body) {
    const text = `${title} ${body}`.toLowerCase();
    const rules = [
      ["旅行", ["旅行", "旅游", "攻略", "行程", "泰国", "曼谷", "普吉", "芭提雅", "京都", "大阪", "东京", "酒店"]],
      ["美食", ["餐厅", "日料", "烤肉", "拉面", "人均", "必点", "菜", "咖啡", "探店"]],
      ["护肤", ["护肤", "防晒", "精华", "面霜", "敏感肌", "成分", "保湿"]],
      ["学习", ["学习", "备考", "考试", "课程", "笔记", "复盘", "效率"]],
    ];
    const matched = rules.find(([, words]) => words.some((word) => text.includes(word)));
    return matched ? matched[0] : "未分类";
  }

  function makeSummary(title, body, category) {
    const text = normalizeSpace(body);
    if (!text) return "暂无摘要";
    const sentences = String(body)
      .split(/[。！？!?\n]+/)
      .map((item) => normalizeSpace(item))
      .filter((item) => item.length >= 8);

    if (category === "美食") {
      const recs = extractRecommendations(body);
      const price = extractPrice(body);
      const parts = [];
      if (recs.length) parts.push(`推荐关注：${recs.slice(0, 3).join("、")}。`);
      if (price) parts.push(`笔记里提到的人均大约 ${price}。`);
      if (parts.length) return parts.join("");
    }

    if (category === "旅行") {
      const places = ["泰国", "曼谷", "普吉", "芭提雅", "日本", "大阪", "京都", "东京", "富士山"].filter((place) => `${title}${body}`.includes(place));
      if (places.length) return `这是一篇关于 ${places.slice(0, 4).join("、")} 的旅行笔记，可重点参考路线、交通和体验安排。`;
    }

    return (sentences.slice(0, 2).join("。") || text).slice(0, 140) + (text.length > 140 ? "..." : "");
  }

  function extractRecommendations(text) {
    const candidates = [];
    const patterns = [
      /(?:必点|推荐|值得点|可以点)[：:\s]*([^。！？!?\n]+)/g,
      /[①②③④⑤⑥⑦⑧⑨⑩]\s*([^：:，,。！？!?\n]+)/g,
      /\d+[.\)、]\s*([^：:，,。！？!?\n]+)/g,
    ];
    patterns.forEach((pattern) => {
      let match;
      while ((match = pattern.exec(text))) {
        match[1].split(/[、，,/\s]+/).forEach((part) => {
          const item = part.trim().replace(/[：:；;。.!！?？]/g, "");
          if (item.length >= 2 && item.length <= 18 && !candidates.includes(item)) candidates.push(item);
        });
      }
    });
    return candidates.slice(0, 5);
  }

  function extractPrice(text) {
    const direct = String(text).match(/(?:人均|预算|价格|花费)[：:\s]*([0-9]+(?:\.[0-9]+)?\s*(?:元|块|rmb|RMB)?)/);
    if (direct) return direct[1].trim().replace(/rmb/i, "元");
    const fuzzy = String(text).match(/(?:人均|预算|价格|花费)[^。！？!?\n]{0,12}?((?:五六十|六七十|四五十|[一二三四五六七八九十百两]+[多几]?(?:十|百)?(?:来)?|[0-9]+)(?:元|块)?)/);
    if (!fuzzy) return "";
    return fuzzy[1].endsWith("元") || fuzzy[1].endsWith("块") ? fuzzy[1] : `${fuzzy[1]}元左右`;
  }

  function readNotes() {
    try {
      const notes = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
      return Array.isArray(notes) ? notes : [];
    } catch (error) {
      return [];
    }
  }

  function writeNotes(notes) {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(notes));
  }

  function tokenize(text) {
    const lowered = String(text || "").toLowerCase();
    const words = lowered.match(/[a-z0-9]+|[\u4e00-\u9fff]{2,}/g) || [];
    const chars = [...lowered].filter((char) => /[\u4e00-\u9fff]/.test(char));
    return [...words, ...chars];
  }

  function scoreNote(note, query) {
    const tokens = tokenize(query);
    const haystack = `${note.title} ${note.category} ${note.summary} ${note.body}`.toLowerCase();
    return tokens.reduce((score, token) => score + (haystack.includes(token) ? 1 : 0), 0);
  }

  function searchNotes(query, limit = 5) {
    return readNotes()
      .map((note) => ({...note, score: scoreNote(note, query)}))
      .filter((note) => note.score > 0)
      .sort((a, b) => b.score - a.score || new Date(b.created_at) - new Date(a.created_at))
      .slice(0, limit);
  }

  function representativeSentences(note, query, limit = 4) {
    const tokens = tokenize(query);
    const sentences = String(note.body || "")
      .split(/[。！？!?\n]+/)
      .map((item) => normalizeSpace(item))
      .filter((item) => item.length >= 8);
    return sentences
      .map((sentence) => ({
        sentence,
        score: tokens.reduce((score, token) => score + (sentence.toLowerCase().includes(token) ? 1 : 0), 0),
      }))
      .sort((a, b) => b.score - a.score || a.sentence.length - b.sentence.length)
      .slice(0, limit)
      .map((item) => item.sentence);
  }

  function makeAnswer(query, matches) {
    if (!matches.length) {
      return "当前本地知识库里没有找到相关内容。你可以先导入相关笔记，再继续提问。";
    }
    const primary = matches[0];
    const queryText = String(query).toLowerCase();
    const recs = extractRecommendations(primary.body);
    const price = extractPrice(primary.body);

    if (/(人均|价格|多少钱|预算)/.test(queryText) && price) {
      return `我主要参考了《${primary.title}》。笔记里提到的人均大约是 ${price}。\n你可以点下方来源回到原文，看具体搭配和上下文。`;
    }

    if (/(必点|推荐|吃什么|点什么|菜)/.test(queryText) && recs.length) {
      return `我主要参考了《${primary.title}》。\n推荐关注：${recs.slice(0, 3).join("、")}。${price ? `\n价格参考：人均大约 ${price}。` : ""}\n下方来源可以点回原文核对。`;
    }

    const points = representativeSentences(primary, query, 4);
    return [
      `我主要参考了《${primary.title}》，本地检索给你的直接结论是：`,
      ...(points.length ? points.map((point, index) => `${index + 1}. ${point}`) : [primary.summary || primary.body.slice(0, 140)]),
      "当前是 GitHub Pages 纯前端版，回答来自浏览器本地检索和模板整理。",
    ].join("\n");
  }

  window.KnowFlowStore = {
    listNotes() {
      return readNotes().sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
    },
    getNote(id) {
      return readNotes().find((note) => String(note.id) === String(id)) || null;
    },
    createNote(payload) {
      const notes = readNotes();
      const title = normalizeSpace(payload.title) || "未命名笔记";
      const body = String(payload.body || "").trim();
      if (!body) throw new Error("body is required");
      const category = normalizeSpace(payload.category) || inferCategory(title, body);
      const chunks = splitText(body);
      const note = {
        id: Date.now(),
        title,
        body,
        category,
        summary: makeSummary(title, body, category),
        created_at: nowISO(),
        source_url: "",
        chunk_count: chunks.length,
        chunks,
      };
      notes.push(note);
      writeNotes(notes);
      return note;
    },
    deleteNote(id) {
      const notes = readNotes();
      const next = notes.filter((note) => String(note.id) !== String(id));
      writeNotes(next);
      return next.length !== notes.length;
    },
    chat(query) {
      const matches = searchNotes(query, 5);
      return {
        answer: makeAnswer(query, matches),
        citations: matches.map((note, index) => ({
          number: index + 1,
          note_id: note.id,
          title: note.title,
          category: note.category,
          text: note.summary,
          chunk_index: 0,
          char_start: 0,
          char_end: 0,
        })),
      };
    },
  };
})();
