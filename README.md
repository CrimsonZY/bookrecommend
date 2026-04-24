# 书籍推荐 Web 服务（BookRecommend）

一个三步式的静态 Web 前端 + FastAPI 推荐后端的书籍推荐系统。前端通过标签化意图（`intent_v2`）与可编辑提示词驱动后端推荐管线，并支持“连续刷新持续排除”（排除历史已推荐过的书，直至后端返回 409 耗尽提示）。

## 功能概览
- **三步式引导**：`#mood-scene` → `#style-more` → `#recommend`
- **意图结构化（v2）**：`mood_tags/scene_tags/style_tags/difficulty/pace/length_type` + 自定义补充文本
- **推荐结果展示**：推荐小卡片 + 详情弹窗（不显示封面）
- **连续刷新**：前端累计 `exclude_book_ids`，后端硬排除后重新推荐；耗尽时返回 **HTTP 409**
- **缓存恢复**：recommend 成功结果会缓存 24h（刷新页面仍可看到结果）

## 技术栈
- **前端**：原生 HTML + CSS + ES Modules（无构建，无 `package.json`）
- **后端**：Python + FastAPI + Uvicorn
- **推荐/算法**：向量召回 + 规则/权重 + 重排 + 多样性选择
- **数据**：`data/books_enriched_v2.json` + `data/book_embeddings_v2.npy` 等冻结资产

## 目录结构（关键）
- `index.html`：前端入口
- `scripts/`：前端逻辑（SPA 路由与视图）
- `styles/`：样式
- `components/`：前端组件
- `api/`：**线上推荐 API（FastAPI，`POST /recommend`）**
- `archive/backend_legacy/backend/`：旧/示例后端（接口与 schema 与当前前端不一致，已归档）
- `ml/`：推荐管线与模型逻辑
- `data/`：书库与向量资产（上线必须带上）
- `config/system_config.json`：推荐系统配置（权重/阈值/资产路径等）

## 本地运行

### 1) 启动后端（推荐接口：`/recommend`）

```bash
python -m pip install -r requirements.txt
python -m uvicorn api.app:app --reload --host 127.0.0.1 --port 8001
```

### 2) 启动前端（静态服务）

```bash
python -m http.server 8000
```

浏览器访问：`http://127.0.0.1:8000/`

## 部署（最快上线方案）

### 架构推荐
- **前端**：Netlify / Vercel（静态托管，发布仓库根目录）
- **后端**：Render / Railway（Python FastAPI）

### 后端（Render / Railway）
- **Build Command**：`pip install -r requirements.txt`
- **Start Command**：`uvicorn api.app:app --host 0.0.0.0 --port $PORT`
- **环境变量**：
  - `ALLOWED_ORIGINS=https://<你的前端域名>`（逗号分隔，可填多个）

### 前端（Netlify / Vercel）
本项目前端无构建步骤，直接托管仓库根目录的静态文件即可。

#### 配置后端地址（必须）
前端默认指向本地 `http://127.0.0.1:8001`。上线时需要在 `index.html` 注入：

```html
<script>
  window.BOOKRECOMMEND_API_BASE = "https://your-backend.example.com";
</script>
```

（该注释模板已在 `index.html` 中提供。）

## 字段与接口约定（摘要）
- **请求（`POST /recommend`）**：`query`, `top_k`, `intent_v2`, `custom_text`, `exclude_book_ids`
- **响应**：`{ success: true, query, books: [...] }`
  - 兼容期内同时提供 legacy 字段（如 `book_id/cover_url/summary`）与 canonical snake_case 字段（如 `subject_id/cover_image/content_intro/details_url/style_tags/...`）

## 面试展示说明
- 前端侧重点：交互引导、动效一致性、可编辑提示词、连续刷新排除与失败/耗尽处理。
- 后端侧重点：无状态刷新排除、召回/重排/多样性选择管线、阈值化耗尽（409）策略、可配置化（`config/system_config.json`）。

