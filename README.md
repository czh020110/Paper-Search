# 总流程

> 当前 README 是方案草案与示例数据流说明。具体模型、阈值、Top-K、批大小、打分权重和部署方案均需通过贯穿式评测、公开测试集与效率评估后锁定。

## 当前实施顺序

1. 基础设施与统一数据契约：先确认配置、API Key、安全、缓存、日志、实验记录、输出目录和统一 Paper 数据模型。
2. 参考系统前置研究：在业务模块实现前吸收 PaSa-7B、SPAR、Ai2 Paper Finder、PaperQA2 的可落地策略。
3. 贯穿式评测基线与输出契约：先建立小型 golden set、统一输出 schema 和实验记录口径，再推进查询理解与检索实现。
4. 查询理解与分解 -> 初检索 -> 粗筛 -> 中筛 -> 精筛 -> 滚雪球 -> 结果整理。
5. 公开测试集回归、效率调优与提交前锁定。

## 示例查询与数据流说明

1. 输入:用户查询  
   “2022 年后，关于大模型幻觉控制的、使用强化学习方法的、在 CVPR 发表的论文“”

2. 前置模块:基础设施、参考系统策略与评测契约

编码前先明确统一 Paper 数据模型、候选池状态、缓存键、日志字段、实验记录格式和 Markdown + nodes/edges 输出 schema；同时建立最小 golden set，用于在查询理解、初检索、滚雪球和筛选阶段持续记录 F1、召回、耗时、API 调用数、Token 成本和缓存命中情况。参考系统研究不作为独立产物滞后处理，而是在业务模块实现前约束查询分解、检索扩展、引文追踪和 Agent/LLM 判定策略。

3. 模块一:查询解构(抽取实体、方法、时间，转化为 API 原生过滤参数), 输出: JSON，使用轻量模型

```json
{
  "original_query": "2022年后，关于大模型幻觉控制的、使用强化学习方法的、在CVPR发表的论文",
  "intent_analysis": {
    "domain": "Computer Vision / Natural Language Processing (Vision-Language Models)",
    "query_type": "semantic",
    "boundary_note": "“2022年后”在检索阶段同时覆盖 >=2022 与 >=2023 两种解释，最终结果阶段再收口"
  },
  "hard_filters": {
    "year": {
      "operator": ">=",
      "value": 2022,
      "relaxed_window": [2022, 2026]
    }
  },
  "ranking_signals": {
    "preferred_venues": [
      "CVPR",
      "IEEE/CVF Conference on Computer Vision and Pattern Recognition",
      "Computer Vision and Pattern Recognition",
      "CVPR Workshop"
    ],
    "venue_match_mode": "fuzzy_match_and_bonus",
    "venue_as_hard_filter": false
  },
  "semantic_queries": {
    "core_concepts": [
      "large language model",
      "LLM",
      "vision-language model",
      "VLM",
      "multimodal model",
      "hallucination",
      "object hallucination",
      "hallucination mitigation",
      "hallucination control"
    ],
    "methodologies": [
      "reinforcement learning",
      "RL",
      "RLHF",
      "RLAIF",
      "preference alignment",
      "reward model"
    ]
  },
  "sub_queries_for_retrieval": [
    "LLM hallucination control reinforcement learning",
    "VLM hallucination mitigation RLHF",
    "object hallucination reinforcement learning vision-language model",
    "multimodal hallucination alignment RLAIF",
    "CVPR hallucination mitigation reward model"
  ],
  "query_expansion_policy": {
    "enabled": true,
    "seed_paper_driven": true,
    "extract_terms_from": ["title", "abstract", "keywords"],
    "max_rounds": 2
  },
  "api_payload_translation": {
    "semantic_scholar": [
      {
        "query": "hallucination reinforcement learning vision-language model",
        "year": "2022-"
      },
      {
        "query": "object hallucination RLHF multimodal model",
        "year": "2022-"
      }
    ],
    "openalex": [
      {
        "search": "hallucination control reinforcement learning",
        "filter": "publication_year:>2021"
      },
      {
        "search": "object hallucination reward model",
        "filter": "publication_year:>2021"
      }
    ]
  }
}
```

其中要区分: `hard_filters` 只保留高可信、易标准化的硬约束。年份可作为硬约束，但需保留**边界放宽**与最终复核机制；`venue` 不再做硬删除条件（各 api 及文章收入对应的同期刊使用的字段可能不同），而是转为 `ranking_signals` 中的强排序信号，采用模糊匹配、别名归一化和摘要/标题证据联合判断。`semantic_queries` 负责 BM25、Embedding 与 query expansion 的语义召回。`intent_analysis.query_type` 明确区分以下三类查询，用于决定后续初检索阶段的检索路由策略：

- **navigational（导航型）**：用户大致知道要找哪篇特定论文，但元信息不完整（例如记得大致标题、作者或关键词）。此类型优先走标题精确匹配路径，首条子查询应尽量接近论文标题或核心短语。
- **semantic（语义型）**：用户想找某类方法、主题或领域的论文，没有锁定具体篇目（例如"关于大模型幻觉控制的论文"）。此类型走多角度语义扩展检索路径，子查询覆盖不同关键词组合和角度。
- **metadata（结构化过滤型）**：用户给出了明确的结构化约束（例如年份、会议、作者、领域等）。此类型优先消费 `api_payload_translation` 中的结构化过滤参数，将硬约束直接翻译为各 API 原生过滤条件。

4. 模块二:初检索, 初次检索不再只使用单条 query 获取 10 篇"种子文献"，而是直接从 `api_payload_translation` 中取 S2 和 OA 各自独立的搜索 payload 分别调用 API。**S2 和 OA 各自独立生成搜索策略，数量和策略可不一致**：S2 偏向语义长句 + 结构化过滤（venue、year、minCitationCount），OA 偏向 BM25 关键词句子 + filter 字段（from_publication_date、type）。`sub_queries_for_retrieval` 仅作为搜索意图摘要，供 Reranker、滚雪球和结果展示使用，不直接作为 API 搜索词。初检索阶段优先保证召回，坏种子交由后续粗筛、Reranker 与 LLM 精筛收口；完成后记录最小 golden set 的召回情况、API 调用数和耗时。与此同时，初检索不再对所有查询使用同一条检索路径，而是根据 `intent_analysis.query_type` 选择不同策略：`navigational` 优先走标题精确匹配路径，`semantic` 优先走多角度语义扩展检索路径，`metadata` 优先消费 `api_payload_translation` 中的结构化过滤参数。

5. 模块三：滚雪球模块 — 在精筛完成后，基于 LLM 判定为”高度相关”的论文自动扩展检索范围。  
   在精筛完成后执行（此时论文已有 LLM 相关性判定），调用 `run_snowball()` 通过 LangGraph StateGraph 迭代执行四个节点：  
   - **query_evolution**: 用 `LLM_FAST_MODEL` 从”高度相关”论文提取 3-5 个未覆盖的新关键词  
   - **re_search**: 新关键词再次调 S2/OA API 补召回，去重后以 `status=”expanded”` 加入候选池，计算重叠率  
   - **expand_citations**: 对高分论文调 S2 references/citations + OA references/citations API，记录 `CitationEdge`  
   - **check_convergence**: 重叠率 >70%、无新关键词、最大轮数(3) 或预算耗尽时停止滚动

- 过滤：每次滚动仅对高可信字段做硬规则过滤，如年份边界、作者唯一限定等；venue 不做硬删除，只记录为强排序信号，交由后续排序和 LLM 终判。
- 去重：优先比对 DOI、ArXiv ID、Semantic Scholar paperId、OpenAlex ID；缺失统一 ID 时再按标题去除标点、空格并转小写后进行归一化字符串匹配。
- 缓存：对论文元数据、引文结果、embedding 结果做缓存，避免同一篇论文在多轮滚动中被重复请求。

6. 模块四：粗筛 — BM25 + Embedding + Structure 三路混合打分

- 当 paper pool 中论文数量 ≤ COARSE_POOL_SKIP_THRESHOLD（默认 40），直接跳过，交给下一个模块（优先保留召回）。
- 三路 min-max 归一化 + 加权融合：
  - 路 A (稀疏检索)：rank-bm25 关键词匹配。权重：COARSE_WEIGHT_BM25=0.35
  - 路 B (稠密检索)：DashScope text-embedding-v4 余弦相似度。权重：COARSE_WEIGHT_EMBEDDING=0.35。硬底线：cosine_sim < COARSE_EMBEDDING_MIN_SIMILARITY=0.35 → 直接排除。并发 10 + Token Bucket 15 RPS 限流。可通过 EMBEDDING_ENABLED=false 关闭
  - 路 C (结构信号)：年份接近度 + log1p(引用数) + venue 模糊命中。权重：COARSE_WEIGHT_STRUCTURE=0.30
- 截断：combined < mean × COARSE_RELATIVE_THRESHOLD_FACTOR=0.3 → excluded
- 无 Embedding 时退化为 0.60×BM25 + 0.40×Structure（权重可配置）
- 所有阈值/权重均为可配置环境变量，Web 配置面板可直接修改

7.  模块五：中筛 — DashScope qwen3-rerank API 远程重排序  
    保底 Top-RERANKER_TOP_K_FALLBACK=30 篇 + 相对阈值 mean × RERANKER_RELATIVE_THRESHOLD_FACTOR=0.7。不设硬编码阈值，参数可配置。无 RERANKER_PROVIDER 配置时 pass-through 全量透传。

8.  模块六：精筛 — LLM 100 并发单篇相关性判定  
    每篇论文独立调用 LLM_FAST_MODEL，ThreadPoolExecutor 100 并发 + 5000 篇/波上限。输出三分类（高度相关/部分相关/不相关）+ 判定理由 + 核心贡献。无 LLM_API_KEY 时报错不静默回退。纳入口径（只收高度相关 vs 同时吸纳部分相关）以 F1 为目标实验调优。  
    LLM 精筛阶段除判断主题相关性外，还负责最终复核年份边界、venue 证据和“是否属于正式结果集”的纳入决策；即使元数据中的 venue 字段缺失，只要标题/摘要/补充字段能够证明论文对应 CVPR 正式发表版本，仍可保留。  
    system prompt 参考：

```markdown
角色：严谨的学术领域专家。根据【原始学术查询】评估以下【候选论文列表】的相关性。

【用户原始学术查询】：
"2022年后，关于大模型幻觉控制的、使用强化学习方法的、在CVPR发表的论文"

【候选论文列表】：
[
  {
    "paper_id": "paper_001",
    "title": "RLHF for Vision-Language Models: Mitigating Hallucinations in Visual Question Answering",
    "abstract": "Large Vision-Language Models (VLMs) often suffer from object hallucination... In this paper, we propose a novel framework using Reinforcement Learning from Human Feedback (RLHF) to align VLM outputs... Published in CVPR 2023.",
    "venue": "arXiv",
    "year": 2023
  }
]

【任务指令】：
1. 对比查询意图与论文内容，判断是否满足主题、方法、时间与 venue 要求。
2. venue 字段缺失或写法不一致时，不直接判负，需结合标题、摘要与其他元数据综合判断。
3. 分类结果只能在 [高度相关, 部分相关, 不相关] 中选择。
4. 输出一句判定理由与一句主要贡献。
5. 严格输出 JSON 数组，不添加额外说明。

输出格式：
[
  {
    "paper_id": "<传入的论文ID>",
    "relevance": "<高度相关/部分相关/不相关>",
    "reason": "<判定理由（一句）>",
    "contribution": "<论文主要贡献（一句）>"
  }
]
```

注：相关性只分三个级别，不使用具体数值分值；最终纳入口径（只收高度相关，或同时吸纳部分相关）需要在公开测试集上以 F1 为目标做实验调优，不能预先写死。

9.  模块七：纯代码 json 整理筛选。不再固定只保留 `relevance = 高度相关`，而是将 `高度相关` 作为主结果集、`部分相关` 作为扩展结果集，最终是否合并输出由公开测试集上的 F1 结果决定。json 格式化整理时加入元数据 `title, abstract, id, author, year, venue, link, reranker_score, llm_relevance, reason, contribution, source_api`。  
    转为 markdown 按键值填充，示例（排序按照综合 score）：

```markdown
### {title} ({year})

- **ID**：{id}
- **作者**：{author}
- **Venue**：{venue}
- **相关度**：{llm_relevance} / {reranker_score}
- **链接**：[访问原文]({link})
- **匹配理由**：{reason}
- **核心贡献**：{contribution}

> **📄 摘要**
> {abstract}

---

......(concat)
```

同时输出关系图所需的结构化数据，例如 `nodes = papers`、`edges = citations/references`，后处理生成引文关系图或小型知识图谱，满足“列表 + 关系图”的结果展示要求。除此之外，每次运行还需要同步产出 `experiment.json`，用于记录 `run_id`、配置快照、阶段指标、API 调用数、Token 成本、缓存命中率和输出文件索引；最终输出契约不再只是 Markdown + `nodes/edges`，而是统一为 `result.md + graph.json + experiment.json` 三类产物。

---

# 使用方式

## 安装

```bash
# 安装 uv（如未安装）
brew install uv

# 在项目根目录初始化虚拟环境并安装依赖
uv sync
```

### 环境配置（重要）

项目使用双 `.env` 机制：

| 文件 | 作用 | 是否提交到 Git |
|------|------|---------------|
| `.env` | **模板文件**，放占位符，给开发者参考 | ✅ 提交 |
| `.env.local` | **本地配置**，放真实 API Key 和模型名 | ❌ 已 .gitignore |

```bash
# 首次使用：复制模板
cp .env .env.local

# 编辑 .env.local 填入真实 API Key
# 程序启动时优先读取 .env.local，未找到时回退到 .env
```

> Web 面板 ⚙ → 点"保存"时自动写入 **.env.local**，不影响模板。`.env` 保留占位符安全提交。

## CLI 使用

### Mock 后端（仅用于开发调试和测试，使用本地 fixtures 数据，无需联网）

> ⚠️ Mock 后端从本地 fixtures 读取硬编码数据，不会调用真实学术 API，返回的论文内容与查询无关。**仅用于开发调试、CI 测试和契约验证，不反映真实检索效果。**

```bash
# 基本用法
uv run paper-search --query "2022年后关于大模型幻觉控制、使用强化学习方法、在CVPR发表的论文" --backend mock

# 指定输出目录
uv run paper-search --query "2022年后关于大模型幻觉控制、使用强化学习方法、在CVPR发表的论文" --backend mock --output-dir outputs

# 也可以使用 python -m 方式调用
uv run python -m paper_search --query "..." --backend mock
```

### Live 后端（默认模式，调用 Semantic Scholar 与 OpenAlex 真实 API）

```bash
# 不指定 --backend 时默认使用 live
uv run paper-search --query "2022年后关于大模型幻觉控制、使用强化学习方法、在CVPR发表的论文"

# 指定输出目录
uv run paper-search --query "..." --backend live --output-dir outputs
```

### Web 界面（固定使用 live 后端）

```bash
# 启动 FastAPI Web 服务（默认端口 8000）
uv run paper-search-web

# 或使用 python -m 方式，可自定义 host 和 port
uv run python -m paper_search --serve --host 127.0.0.1 --port 8000

# 浏览器访问 http://127.0.0.1:8000，输入查询即可检索真实学术 API
```

启动后可访问以下页面：

| 地址 | 说明 |
|------|------|
| `http://127.0.0.1:8000/search` | 搜索页面，输入自然语言查询（实时进度条 + 耗时显示 + 悬浮详情） |
| `http://127.0.0.1:8000/docs` | **Swagger UI** — 交互式 API 调试界面，可直接发送请求并查看响应 |
| `http://127.0.0.1:8000/redoc` | **ReDoc** — API 文档阅读界面 |

API 端点：

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/search?q=...` | 检索论文 |
| GET | `/api/search/stream?q=...` | **SSE 流式检索** — 实时推送进度（keys/progress/result）和模块耗时 |
| POST | `/api/search` | 检索论文（JSON body: `{"query": "..."}`） |
| GET | `/api/status` | 返回各 API Key 配置状态（llm/s2/embedding/reranker）及 config_ready 完整性检查 |
| GET | `/api/config` | 获取所有可配置项及其当前值、帮助文本和必需标记 |
| POST | `/api/config` | 更新配置（写入 `.env.local`），保存后即时生效 |
| GET | `/api/config/validate` | 返回配置完整性检查结果（ready + missing 列表） |
| GET | `/api/runs` | 列出历史运行 |
| GET | `/api/runs/{run_id}` | 获取某次运行详情 |
| GET | `/api/runs/{run_id}/experiment` | 获取实验记录 JSON |

## 输出产物

每次运行会在 `outputs/<run_id>/` 下生成：

| 文件 | 内容 |
|------|------|
| `result.md` | Markdown 论文列表，含查询摘要、高度相关论文、部分相关论文、引文关系说明和运行摘要；每篇论文显示相关度（含 reranker_score）、匹配理由、核心贡献、摘要 |
| `graph.json` | 结构化数据，含 `query`、`nodes`（论文节点列表）、`edges`（引文关系边列表） |
| `experiment.json` | 实验记录，含 `run_id`、配置快照、阶段指标（query_understanding / initial_retrieval / snowball / coarse / rerank / judge / result_format / overall）、各模块耗时（stage_timings, ms）、预算使用、输出文件路径 |
| `query_plan.json` | 完整 QueryPlan JSON 持久化，含 query_type、hard_filters、ranking_signals、semantic_queries、sub_queries_for_retrieval、api_payload_translation |
| `logs/pipeline.jsonl` | 结构化 JSON 日志，每行一条记录，含 timestamp、level、logger、message、run_id、stage 等字段 |

## 运行测试

```bash
# unittest（79 个测试）
uv run python -m unittest discover -s tests -v

# pytest
uv run python -m pytest tests/ -v
```

## 环境变量说明

| 变量 | 必需 | 说明 |
|------|------|------|
| `LLM_PROVIDER` | 否 | LLM 提供商：`openai`（OpenAI 兼容）或 `dashscope`；默认 `openai` |
| `LLM_MODEL` | 否 | LLM 模型名称，默认 `gpt-4o-mini` |
| `LLM_FAST_MODEL` | 否 | 轻量/快速 LLM 模型名称，用于查询理解和精筛判定，默认回退到 `LLM_MODEL` |
| `LLM_API_KEY` | 是（live 模式） | LLM API 认证密钥，查询理解和精筛模块均依赖；live 模式下未配置时报错 |
| `OPENAI_BASE_URL` | 否 | OpenAI 兼容 API 端点，默认 `https://api.openai.com/v1` |
| `LLM_THINKING` | 否 | 思考/推理深度：`none`(关闭,默认)/`off`(不传参)/`minimal`/`low`/`medium`/`high`/`xhigh` |
| `SEMANTIC_SCHOLAR_API_KEY` | 否 | Semantic Scholar API 认证，有 key 可提升速率限制 |
| `OPENALEX_MAILTO` | 否 | OpenAlex 礼貌池参数，填入邮箱可获得更稳定服务 |
| `EMBEDDING_PROVIDER` | 否 | Embedding 服务提供商，如 `dashscope` |
| `EMBEDDING_MODEL` | 否 | Embedding 模型，如 `text-embedding-v4` |
| `EMBEDDING_API_KEY` | 否 | Embedding API 认证密钥 |
| `EMBEDDING_ENABLED` | 否 | 是否启用 Embedding，`true`/`false`；默认 `true` |
| `EMBEDDING_CONCURRENCY` | 否 | Embedding 并发 worker 数；默认 `10` |
| `EMBEDDING_RPS_LIMIT` | 否 | Embedding API 每秒最大请求数；默认 `15` |
| `RERANKER_PROVIDER` | 否 | Reranker 服务提供商，如 `dashscope` |
| `RERANKER_MODEL` | 否 | Reranker 模型，如 `qwen3-rerank` |
| `RERANKER_API_KEY` | 否 | Reranker API 认证密钥 |
| `COARSE_POOL_SKIP_THRESHOLD` | 否 | 候选池 ≤N 跳过粗筛；默认 `40` |
| `COARSE_EMBEDDING_MIN_SIMILARITY` | 否 | Embedding 硬底线；默认 `0.35` |
| `COARSE_WEIGHT_BM25` | 否 | BM25 三路权重；默认 `0.35` |
| `COARSE_WEIGHT_EMBEDDING` | 否 | Embedding 三路权重；默认 `0.35` |
| `COARSE_WEIGHT_STRUCTURE` | 否 | Structure 三路权重；默认 `0.30` |
| `COARSE_WEIGHT_BM25_FALLBACK` | 否 | 降级 BM25 权重；默认 `0.60` |
| `COARSE_WEIGHT_STRUCTURE_FALLBACK` | 否 | 降级 Structure 权重；默认 `0.40` |
| `COARSE_RELATIVE_THRESHOLD_FACTOR` | 否 | 截断因子；默认 `0.3` |
| `RERANKER_TOP_K_FALLBACK` | 否 | 重排序 Top-K 保底；默认 `30` |
| `RERANKER_RELATIVE_THRESHOLD_FACTOR` | 否 | 重排序阈值因子；默认 `0.7` |
| `JUDGE_CONCURRENCY` | 否 | 精筛并发数；默认 `100` |
| `JUDGE_WAVE_CAP` | 否 | 精筛单波上限；默认 `5000` |
| `CACHE_DIR` | 否 | 缓存根目录，默认 `.cache` |
| `OUTPUT_DIR` | 否 | 结果输出根目录，默认 `outputs` |
| `LOG_LEVEL` | 否 | 日志级别，默认 `INFO` |

> 配置写入 `.env.local`（不提交 Git）。`.env` 保留为占位符模板。Web 面板也可直接修改。

---

# 方向

# STEP

## 长期方向
- [x] S-000：项目基础设施与配置骨架确认（硬前置）
  - 目标：先锁定端到端流程所需的配置管理、API Key 管理、安全原则、缓存目录、日志策略、实验记录、结果输出目录和统一 Paper 数据模型
  - 完成标准：形成可指导后续代码实现的基础设施约定；明确 Paper 元数据字段、候选池状态、缓存键、日志字段、实验记录格式和输出目录；不在文档阶段创建代码脚手架或锁定最终部署方案
  - 影响范围：后续所有模块的运行配置、输入输出契约、可复现实验和安全边界

- [x] S-001：参考系统研究与策略对齐
  - 目标：在业务模块实现前短周期研究 PaSa-7B、SPAR、Ai2 Paper Finder、PaperQA2 等参考系统，提取可借鉴的查询分解、检索扩展、引文追踪、Agent 协作和评估策略
  - 完成标准：形成可落地的策略取舍，并将有效做法映射到查询理解、检索扩展、筛选判定、结果展示和评估基线中
  - 影响范围：系统方案创新性、落地可行性和算法泛化性，减少后续检索策略返工

- [x] S-002：贯穿式评测基线与输出契约建立
  - 目标：在查询理解和初检索实现前建立小型 golden set、统一输出 schema、评测指标口径和实验记录方式，让后续模型、阈值、Top-K、批大小和预算策略可比较
  - 完成标准：明确最小评测样例、期望论文集合标注口径、F1/耗时/API 调用数/Token 成本/缓存命中率记录字段，以及 Markdown + nodes/edges 输出 schema；公开测试集接入后沿用同一评估框架扩展
  - 影响范围：S-003 之后所有模块的参数校准、回归测试和提交前复现实验

- [x] S-003：查询理解与分解模块实现（live 模式用 Pydantic `with_structured_output` + LLM，mock 用规则）
  - 目标：实现自然语言学术查询到结构化检索参数的自动转化，含查询类型识别（navigational/semantic/metadata）、子查询分解与查询改写扩展
  - 完成标准：live 模式通过 LangChain `with_structured_output` + Pydantic `QueryPlanSchema` 调用 LLM_FAST_MODEL，类型/数量/结构约束由 Pydantic 承担，跨字段规则由精简 prompt 承担。输出包含 query_type、hard_filters、ranking_signals（preferred_venues 为 list[str]）、semantic_queries（core_concepts + methodologies）、sub_queries_for_retrieval、api_payload_translation、query_expansion_policy；无 LLM_API_KEY 直接报错
  - 影响范围：后续所有检索模块的输入依赖

- [x] S-004：初检索模块实现（S2+OA 双源并行）
- [x] S-003.5：公共服务底座打通（缓存/预算/状态机/日志）
- [x] S-005：滚雪球模块实现（LLM query evolution + S2/OA 引文扩展 + LangGraph 迭代）
- [ ] S-005.5：双模式运行（fast / exhaustive）
- [x] S-006：粗筛模块实现（BM25 + DashScope Embedding + Structure 三路融合，≤40 跳过，Embedding 硬底线 0.35，三路权重 0.35/0.35/0.30）
- [x] S-007：中筛模块实现（qwen3-rerank API 远程重排，相对阈值 mean×0.7 + Top-30 保底）
- [x] S-008：精筛模块实现（LLM 100 并发单篇判定，三分类：高度相关/部分相关/不相关 + 理由 + 贡献）
- [x] S-009：结果整理模块实现（result.md + graph.json + experiment.json + query_plan.json + logs/）
- [ ] S-010：公开测试集回归、效率调优与提交前锁定
- [ ] S-011：README 与项目文档同步维护
- [ ] S-011.5：CLI 子命令重构（run / eval / replay / inspect）