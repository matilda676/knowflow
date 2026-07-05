# KnowFlow 公网部署说明

目标：让任意手机都能通过公网链接访问 KnowFlow，而不是只能在同一个 Wi-Fi 下访问。

## 推荐方式

优先用支持 Python Web Service 和持久化磁盘的平台，比如 Render、Railway、Fly.io，或阿里云/腾讯云服务器。

本项目已经准备好：

- 云端启动命令：`gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120`
- 启动文件：`Procfile`
- 数据目录环境变量：`KNOWFLOW_DATA_DIR`
- 页面访问密码：`KNOWFLOW_ACCESS_PASSWORD`
- 大模型环境变量：`KNOWFLOW_LLM_API_KEY`、`KNOWFLOW_LLM_BASE_URL`、`KNOWFLOW_LLM_MODEL`

## 必填环境变量

```bash
KNOWFLOW_ACCESS_PASSWORD=自己设置一个访问密码
KNOWFLOW_DATA_DIR=/data
```

访问时用户名固定填：

```text
knowflow
```

密码就是你设置的 `KNOWFLOW_ACCESS_PASSWORD`。

## 可选大模型变量

如果使用 OpenAI 兼容接口：

```bash
KNOWFLOW_LLM_API_KEY=你的 API key
KNOWFLOW_LLM_BASE_URL=https://api.deepseek.com
KNOWFLOW_LLM_MODEL=deepseek-chat
```

如果不配置，系统会继续使用本地 fallback，能演示，但 AI 回答和摘要质量有限。

## 可选语义检索变量

如果云平台资源允许，可以安装 `requirements-embedding.txt`，并设置：

```bash
KNOWFLOW_EMBEDDING_PROVIDER=sentence-transformers
KNOWFLOW_EMBEDDING_MODEL=shibing624/text2vec-base-chinese
```

免费小机器可能装不动 `torch`，这种情况下先不装也可以，核心流程仍能跑。

## 部署后检查

打开公网链接后，如果浏览器弹用户名密码：

- 用户名：`knowflow`
- 密码：你设置的访问密码

进入后检查：

- `/api/health` 返回 `ok: true`
- 知识库能读取内容
- 导入新内容后，刷新页面仍然存在
- 聊天引用来源能点回原文

## 重要提醒

公网部署后，不要把 API key 写进代码或提交到仓库。所有 key 都只放到平台的环境变量里。
