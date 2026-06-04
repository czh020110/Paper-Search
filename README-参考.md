# 总流程

> 当前 README 是方案草案与示例数据流说明。具体模型、阈值、Top-K、批大小、打分权重和部署方案均需通过贯穿式评测、公开测试集与效率评估后锁定。

## 当前实施顺序

1. 基础设施与统一数据契约：先确认配置、API Key、安全、缓存、日志、实验记录、输出目录和统一 Paper 数据模型。
2. 参考系统前置研究：在业务模块实现前吸收 PaSa-7B、SPAR、Ai2 Paper Finder、PaperQA2 的可落地策略。
3. 贯穿式评测基线与输出契约：先建立小型 golden set、统一输出 schema 和实验记录口径，再推进查询理解与检索实现。
4. 查询理解与分解 -> 初检索 -> 滚雪球 -> 粗筛 -> 中筛 -> 精筛 -> 结果整理。
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

4. 模块二:初检索, 初次检索不再只使用单条 query 获取 10 篇"种子文献"，而是对 `sub_queries_for_retrieval` 逐条调用 Semantic Scholar 与 OpenAlex 两家学术论文检索 API（Semantic Scholar 侧重计算机科学和生物医学领域的语义搜索，OpenAlex 为开放学术索引覆盖面广），每个 query × 每个 API 各取一批结果，合并去重后形成 30~50 篇"种子文献"候选池。初检索阶段优先保证召回，坏种子交由后续粗筛、Reranker 与 LLM 精筛收口；完成后记录最小 golden set 的召回情况、API 调用数和耗时。与此同时，初检索不再对所有查询使用同一条检索路径，而是根据 `intent_analysis.query_type` 选择不同策略：`navigational` 优先走标题精确匹配路径，`semantic` 优先走多角度语义扩展检索路径，`metadata` 优先消费 `api_payload_translation` 中的结构化过滤参数；`api_payload_translation` 在初检索阶段必须被实际使用，而不是只作为中间产物生成后闲置。

5. 模块三：滚雪球模块:调用 api 获取种子论文的 References(引用)和 Citations(被引用), 将这些论文加入候选池. （提高召回）  
   滚雪球不再只做无差别引文扩展，而是加入“LLM 驱动 query evolution + 选择性引文扩展”的迭代式检索闭环。具体做法如下：先根据初检索高分论文的标题、摘要和关键词，抽取第二轮检索词（如 `object hallucination`、`reward model`、`preference alignment` 等），回到检索 API 再做一轮补召回；随后仅对 Reranker（一种基于深度学习的精细排序模型，以原始查询与论文标题/摘要为输入，输出每篇论文的精准相关度分数）或 LLM 判为高相关或部分相关、且分数靠前的论文继续展开 References/Citations，按相关度优先级滚动扩展，而不是对全部种子无差别展开。  
   滚雪球需要多次滚动，每次通过 API 拉取一批新文献后，立刻与现有候选池（Paper Pool）做集合求交（基于 DOI、ArXiv ID、Semantic Scholar paperId、OpenAlex ID 或归一化 Title）。当新增论文与候选池重叠超过 70%、连续一轮没有新增高价值关键词、达到最大轮数（2~3 轮）或触发预算上限（API 调用数 / 候选池上限 / 单篇最大扩展数）时停止滚动。过程中发现的引文关系（谁引用了谁、谁被谁引用）记录为 CitationEdge（包含 source_paper_id、target_paper_id、edge_type、discovered_round 等字段），供后续 graph.json 的 edges 数据使用。

- 过滤：每次滚动仅对高可信字段做硬规则过滤，如年份边界、作者唯一限定等；venue 不做硬删除，只记录为强排序信号，交由后续排序和 LLM 终判。
- 去重：优先比对 DOI、ArXiv ID、Semantic Scholar paperId、OpenAlex ID；缺失统一 ID 时再按标题去除标点、空格并转小写后进行归一化字符串匹配。
- 缓存：对论文元数据、引文结果、embedding 结果做缓存，避免同一篇论文在多轮滚动中被重复请求。

6. 模块四：粗筛，直接根据 API 关联度得分、刊物信号、BM25 关键词匹配算法、嵌入模型得出余弦相似度

- 当 paper pool 中论文数量 <= 80，直接跳过，交给下一个模块（优先保留召回）；该数量阈值需通过公开测试集评估后确认。
- 混合打分：
  - 路 A (稀疏检索)：BM25 计算关键词匹配得分。
  - 路 B (稠密检索)：候选轻量级 Embedding 模型（如 `BAAI/bge-small-zh-v1.5` 或同级英文/多语模型）计算余弦相似度，最终选择需根据实际语种分布与公开测试集效果确认。
  - 路 C (结构信号)：API 原始相关度、venue 模糊命中、引文图邻接强度、是否由高相关种子扩展而来。  
    加权融合与截断：将多路分数归一化后加权求和，BM25、Embedding、Structure 权重作为待评估参数；按总分降序排列，但不再固定严格截取 Top 50，而是采用 `Top-K + 相对阈值` 的候选策略，尽量避免过早截断召回。粗筛完成后必须记录召回损失、候选规模变化和耗时变化。

7.  模块五：中筛，使用重排序模型 Reranker 进行高精度排序得分。候选模型包括高精度本地模型 `BAAI/bge-reranker-v2-gemma` 与均衡模型 `BAAI/bge-reranker-v2-m3`，最终选择、是否分阶段精排以及候选池大小阈值需通过公开测试集与本地资源评估后确认。  
    不使用固定 `0.7` 这类硬编码阈值，而采用“相对阈值 + Top-K 保底”策略；Top-K 与 α 等参数需根据不同查询分布实验校准。

8.  模块六：精筛，使用 LLM 精确筛选。执行方式候选为“每批 5~10 篇论文一次调用”的批处理模式，用于降低 token 与请求成本；批大小、并发量和模型版本需通过公开测试集与效率评估锁定。候选模型可包含 `qwen3.7 max`、`deepseek v4`、`glm5.1`、`minimax2.7` 等，提交前再按实际部署环境锁定具体模型版本、计费方式与吞吐性能。  
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

# 方向

# STEP

## 长期方向
- [x] S-000：项目基础设施与配置骨架确认（硬前置）
  - 目标：先锁定端到端流程所需的配置管理、API Key 管理、安全原则、缓存目录、日志策略、实验记录、结果输出目录和统一 Paper 数据模型
  - 完成标准：形成可指导后续代码实现的基础设施约定；明确 Paper 元数据字段、候选池状态、缓存键、日志字段、实验记录格式和输出目录；不在文档阶段创建代码脚手架或锁定最终部署方案
  - 影响范围：后续所有模块的运行配置、输入输出契约、可复现实验和安全边界

- [ ] S-001：参考系统研究与策略对齐
  - 目标：在业务模块实现前短周期研究 PaSa-7B、SPAR、Ai2 Paper Finder、PaperQA2 等参考系统，提取可借鉴的查询分解、检索扩展、引文追踪、Agent 协作和评估策略
  - 完成标准：形成可落地的策略取舍，并将有效做法映射到查询理解、检索扩展、筛选判定、结果展示和评估基线中
  - 影响范围：系统方案创新性、落地可行性和算法泛化性，减少后续检索策略返工

- [ ] S-002：贯穿式评测基线与输出契约建立
  - 目标：在查询理解和初检索实现前建立小型 golden set、统一输出 schema、评测指标口径和实验记录方式，让后续模型、阈值、Top-K、批大小和预算策略可比较
  - 完成标准：明确最小评测样例、期望论文集合标注口径、F1/耗时/API 调用数/Token 成本/缓存命中率记录字段，以及 Markdown + nodes/edges 输出 schema；公开测试集接入后沿用同一评估框架扩展
  - 影响范围：S-003 之后所有模块的参数校准、回归测试和提交前复现实验

- [ ] S-003：查询理解与分解模块实现
  - 目标：实现自然语言学术查询到结构化检索参数的自动转化，含查询类型识别（navigational/semantic/metadata）、子查询分解与查询改写扩展
  - 完成标准：输入自然语言查询，输出包含 query_type、hard_filters、ranking_signals、semantic_queries、sub_queries_for_retrieval、api_payload_translation 的完整 JSON；query_type 决定后续检索策略路由；字段结构和可调参数需支持贯穿式评测与公开测试集校准
  - 影响范围：后续所有检索模块的输入依赖

- [ ] S-004：初检索模块实现
  - 目标：多源 API 并行检索，根据查询类型路由检索策略，实际使用 api_payload_translation 定制参数，形成种子文献候选池
  - 完成标准：对 sub_queries_for_retrieval 逐条调用 Semantic Scholar 与 OpenAlex，navigational 查询走标题精确匹配路径，metadata 查询优先使用 api_payload_translation，合并去重后候选池 30~50 篇；单 query 返回数量、API 调用预算和并发策略作为待评估参数；完成后跑最小 golden set 召回检查
  - 影响范围：滚雪球模块的输入

- [ ] S-005：滚雪球模块实现
  - 目标：基于引文图的选择性迭代扩展与自动收敛，集成查询演化（Query Evolution）
  - 完成标准：对精筛后高度相关论文（采用 RCS 式评估，即 1-10 分相关性评分 + 摘要重写，8 分及以上触发引文追踪）做双向引文追踪（1层深度），从相关论文中提取新术语做查询演化；多轮滚动后候选池有效扩展，达到收敛条件时自动停止；收敛条件含重叠率>70%、连续2轮无新关键词、最大轮数或预算耗尽；记录引文关系边到 CitationEdge（引文关系边数据结构，含来源论文ID、目标论文ID、边类型、发现轮次等字段）
  - 影响范围：粗筛模块的输入

- [ ] S-006：粗筛模块实现
  - 目标：BM25 + Embedding + 结构信号三路混合打分与截断
  - 完成标准：候选池>80 时输出截断后候选集，<=80 时跳过；三路混合打分 coarse_total = w1*bm25 + w2*embedding + w3*structure；权重及阈值均为待评估参数；粗筛结果保留分数供后续阶段参考；完成后记录召回损失与耗时变化
  - 影响范围：中筛模块的输入

- [ ] S-007：中筛模块实现
  - 目标：Reranker 高精度重排序，相对阈值 + Top-K 保底
  - 完成标准：输出重排序后候选集与 reranker_score，取高于均值的论文保底取 Top-30；Reranker 模型、Top-K 和相对阈值通过贯穿式评测与公开测试集锁定
  - 影响范围：精筛模块的输入

- [ ] S-008：精筛模块实现
  - 目标：LLM 批处理相关性判定，采用 RCS 式评估（1-10 分评分 + 摘要重写）
  - 完成标准：按批输出 1-10 相关性评分 + 重写摘要 + 判定理由的 JSON；RCS≥8 分的论文触发滚雪球引文追踪反馈；批大小、模型版本、纳入口径和 Token 预算通过贯穿式评测、公开测试集与效率评估确定
  - 影响范围：结果整理模块的输入

- [ ] S-009：结果整理模块实现
  - 目标：Markdown 论文列表 + 引文关系图结构化数据输出，满足赛事结构化展示要求
  - 完成标准：输出格式规范的 Markdown 与 nodes/edges 结构化数据，包含列表与关系图双形式展示；输出结构需包含 nodes（论文ID、标题、年份、venue、相关度、来源API）和 edges（引文关系：来源论文 → 目标论文及边类型），与贯穿式评测输出 schema 保持一致
  - 影响范围：最终用户可见输出，直接影响评测中"回复结果结构化"10% 权重得分

- [ ] S-010：公开测试集回归、效率调优与提交前锁定
  - 目标：基于公开测试集和前期 golden set 评估 F1、端到端耗时、API 调用数、Token 成本和缓存命中情况，校准召回、筛选和输出参数
  - 完成标准：形成可复现实验结果，锁定提交前的模型、阈值、Top-K、批大小、预算策略和结果输出口径
  - 影响范围：最终竞赛得分中的 F1 Score、运行效率和结构化输出质量

- [ ] S-011：README 与项目文档同步维护
  - 目标：当项目规划或说明文档发生实质变化时，执行同步检查
  - 完成标准：README 仅表达当前方案草案和文档索引，不把待评估模型、阈值、权重或部署方案写成最终决策
  - 影响范围：项目说明一致性和后续协作可维护性