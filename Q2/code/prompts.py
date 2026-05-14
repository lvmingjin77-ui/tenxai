"""Task prompts for each phase of the feature engineering agent."""

DIAGNOSE = """
当前共享状态：
- state['df']: 原始面板数据 (81046, 321)，含X1~X300特征
- state['x_cols']: 特征名列表（300个）
- state['target']: 目标变量 'Y7'

任务：对全部300个X特征做全面诊断，将结果保存到 state['diagnose_report']。

请生成代码（代码中可 import 任何需要的库）完成：
1. 对每个特征计算：null_pct, zero_pct, mean, std, skew, kurt, outlier_pct（IQR法）
2. 检测缺失模式：缺失行索引完全一致的特征归为同一 null_group（用frozenset hash标记）
3. 打印摘要（各null_group特征数，分布类型统计：|skew|<1为正态，1~5为偏态，>5为重尾）
4. 将DataFrame保存到 state['diagnose_report']

数据通过 state['df'] 访问，结果写回 state。
"""

CLEAN = """
当前共享状态：
- state['df']: 原始数据
- state['diagnose_report']: 诊断报告（包含null_pct, skew等列）
- state['x_cols']: 特征名列表
- state['train_dates']: 训练集日期（set类型）

任务：清理特征，严防数据泄漏（所有统计量只在训练集上计算）。

请按顺序执行：
1. 缺失值填充：
   - 先排序 df，对每个 underlying 做 ffill（前向填充）
   - 剩余缺失：用训练集中该 trade_date 的截面中位数填充
2. 异常值 clip（只用训练集统计量）：
   - |skew| > 5 的特征：用训练集 [1%, 99%] 分位截断
   - |skew| <= 5：用训练集 mean ± 3*std 截断
3. 截面 rank 归一化：每个 trade_date 内，所有 x_cols 做 rank(pct=True) → [0,1]
4. 将清理结果保存到 state['df_clean']
5. 打印每步处理的特征数和用时

用 log_decision 记录各类处理的决策理由。
"""

EVALUATE = """
当前共享状态：
- state['eval_df']: 已计算完成的特征评估表（columns: feature, ic_mean, ic_std, ic_ir, auc, monotonicity）
- state['corr_matrix']: 训练集上特征间 Spearman 相关矩阵

任务：分析评估结果，输出总体判断和筛选策略建议。

请：
1. 打印 eval_df 的 describe()
2. 统计 |ic_mean| 分布：>0.05, >0.03, 0.01~0.03, <0.01 的特征数
3. 分析相关矩阵：|corr|>0.8 的特征对数
4. 给出综合判断：
   - 信号质量总体评价
   - 推荐的 IC 过滤阈值
   - 去冗余策略建议
5. 将总结文字保存到 state['eval_summary']
"""

SELECT = """
当前共享状态：
- state['eval_df']: 特征评估表（feature, ic_mean, ic_std, ic_ir, auc, monotonicity）
- state['corr_matrix']: 相关矩阵（index/columns 为特征名）
- state['eval_summary']: 你的评估总结

任务：多轮漏斗筛选，从300个特征中选出最优50个。

请逐轮执行并打印每轮特征数：
- Round 1 基础过滤：删除 (|ic_mean| < 0.005 AND auc < 0.51) 的特征（两条件同时满足才删）
- Round 2 综合评分：score = 0.4*rank(|ic_mean|) + 0.3*rank(|ic_ir|) + 0.3*rank(auc)，取 Top 100
- Round 3 去冗余：对 Round 2 剩余特征，|corr| > 0.8 的聚为一组，每组只保留 score 最高的
- Round 4 最终截断：按 score 降序取 Top 50

关键约束（必须满足）：
1) top50 必须是 score 从高到低的前50个（禁止按升序取分数）
2) top50 中每个特征都必须来自 eval_df['feature']
3) top50 必须去重后数量=50
4) 输出并检查质量：
   - top50 的 mean(|ic_mean|) 应明显高于全集均值
   - top50 与 |ic_mean| 前100特征应有明显重合
若不满足，请自行修正后再写回 state。

保存：
- state['top50']: list，最终50个特征名
- state['select_log']: dict，{'r1': N, 'r2': N, 'r3': N, 'r4': N}
- state['top50_scores']: DataFrame，含 feature, ic_mean, ic_ir, auc, score

打印最终 Top 50 及核心指标。
"""
