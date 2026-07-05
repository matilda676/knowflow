# KnowFlow LLM 生成层

KnowFlow 已支持 OpenAI-compatible 的聊天生成接口。

## 基础配置

```bash
export OPENAI_API_KEY="你的 key"
```

可选配置：

```bash
export KNOWFLOW_LLM_BASE_URL="https://api.openai.com/v1"
export KNOWFLOW_LLM_MODEL="gpt-4o-mini"
```

如果使用 DeepSeek：

```bash
export KNOWFLOW_LLM_API_KEY="你的 DeepSeek API key"
export KNOWFLOW_LLM_BASE_URL="https://api.deepseek.com"
export KNOWFLOW_LLM_MODEL="deepseek-v4-flash"
```

DeepSeek 官方当前推荐模型包括 `deepseek-v4-flash` 和 `deepseek-v4-pro`。旧模型名 `deepseek-chat` / `deepseek-reasoner` 将在 2026-07-24 废弃。

如果使用 OpenRouter 或其他兼容 `/chat/completions` 的服务，把 `KNOWFLOW_LLM_BASE_URL` 和 `KNOWFLOW_LLM_MODEL` 换成对应值即可。

## 当前行为

- 未配置 key：继续使用本地规则摘要和回答，保证演示可跑。
- 已配置 key：聊天回答走 RAG + LLM，新导入笔记摘要也会优先走 LLM。
- 检索仍由 ChromaDB + 中文 embedding 完成。

检查状态：

```bash
curl http://127.0.0.1:5001/api/health
```

返回里的 `llm.ready` 为 `true` 时，说明生成层已经接上。
