# 豆瓣读书 Top250 爬虫（前 100）

本脚本使用 `requests` + `BeautifulSoup` 抓取豆瓣读书 Top250 列表中的前 100 本书的**非用户生成内容**字段，并输出到 `data/` 目录下的 CSV/JSON。

## 安装依赖

在项目根目录执行：

```bash
python -m pip install -r scraper/requirements.txt
```

## 运行

默认抓取前 100 本，输出 CSV+JSON：

```bash
python scraper/douban_scraper.py --limit 100 --out data --format both
```

### 常用参数

- `--limit`：抓取条数（1-250，默认 100）
- `--format`：`csv` / `json` / `both`
- `--sleep-min`、`--sleep-max`：每次请求后的随机等待秒数（礼貌抓取）
- `--timeout-connect`、`--timeout-read`：超时（秒）
- `--retries`：对 429/5xx 或被识别为验证页时的重试次数

## 输出字段（统一 schema）

- `subject_id`
- `title`
- `author`
- `author_era_nationality`（页面出现如 `[清]`、`［英］` 等才会有）
- `publisher`
- `content_intro`
- `translator`（若无则为空）
- `pages`（尽量提取数字；否则为空或原文本）
- `cover_image`
- `details_url`

