"""AstaBench PaperFinder 评测模块。

下载 AstaBench paper_finder_bench 数据集，跑我们的搜索管线，
计算 recall@K / precision / F1，与官方排行榜对比。

目前仅支持 specific / metadata 类型（ground truth 为 corpus_ids）。
semantic 类型需要 LLM 评判，暂不支持。
"""
