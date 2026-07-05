# KnowFlow Embedding 方案

## 当前实现

KnowFlow 的向量检索已经接入 ChromaDB。Embedding 层使用可插拔 Provider：

1. 优先尝试 `sentence-transformers`
2. 如果本机没有安装模型依赖，自动回退到 `local-hash-embedding`

这样可以保证 MVP 随时可跑，同时给后续替换专业中文向量模型留好接口。

## 推荐中文模型

默认模型名：

```text
shibing624/text2vec-base-chinese
```

它适合中文语义检索，比当前 fallback 的哈希向量更懂“意思相近”的表达。

## 切换方式

安装依赖：

```bash
source .venv/bin/activate
pip install -r requirements-embedding.txt
```

可选环境变量：

```bash
export KNOWFLOW_EMBEDDING_PROVIDER=sentence-transformers
export KNOWFLOW_EMBEDDING_MODEL=shibing624/text2vec-base-chinese
```

模型文件默认会缓存到项目内的 `data/model_cache`，避免写入系统用户缓存目录导致权限问题。

重建索引：

```bash
python rag/rebuild_index.py
```

或者在服务启动后调用接口：

```bash
curl -X POST http://127.0.0.1:5001/api/reindex
```

检查当前 provider：

```bash
curl http://127.0.0.1:5001/api/health
```

如果返回：

```json
"model_ready": true
```

说明已经使用真实 embedding 模型。如果是 `false`，说明当前仍在使用本地 fallback。
