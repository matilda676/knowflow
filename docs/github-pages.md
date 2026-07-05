# KnowFlow GitHub Pages 纯前端版

这个版本不需要服务器，可以直接放到 GitHub Pages 上访问。

## 怎么开启

1. 打开 GitHub 仓库 `matilda676/knowflow`
2. 进入 `Settings`
3. 左侧选择 `Pages`
4. `Build and deployment` 选择：
   - Source: `Deploy from a branch`
   - Branch: `main`
   - Folder: `/root`
5. 保存后等待 GitHub Pages 构建完成

完成后访问：

```text
https://matilda676.github.io/knowflow/
```

## 这个版本能做什么

- 手机和电脑都能打开
- 导入笔记
- 自动/手动分类
- 知识库列表
- 详情页
- 本地检索式聊天回答

## 数据保存在哪里

数据保存在当前浏览器的 `localStorage` 里。

这意味着：

- 同一台手机、同一个浏览器里可以持续使用
- 换手机或换浏览器后，数据不会自动同步
- 清理浏览器数据后，知识库也会被清空

## 这个版本没有什么

- 没有 Flask 后端
- 没有 SQLite 云端数据库
- 没有 ChromaDB 向量检索
- 没有真正大模型回答
- 不适合保存特别重要的数据

它适合用来先验证产品流程和手机体验。等流程稳定后，再升级回云服务器后端版。
