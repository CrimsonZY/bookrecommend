# 后端（FastAPI）

## 启动（已归档）

此目录为旧版/示例后端，不再作为项目正式后端入口。当前项目正式后端为仓库根目录的 `api/`（`uvicorn api.app:app ...`）。

如需本地复现该旧接口，请在仓库根目录执行：

```bash
python -m pip install -r archive/backend_legacy/backend/requirements.txt
python -m uvicorn archive.backend_legacy.backend.main:app --reload --port 8000
```

## 接口

- `POST /api/recommend`
  - 请求体：`{ "prompt": "..." }`
  - 响应体：
    - `status`: `"success" | "error"`
    - `message`: string
    - `data`: 书籍数组

