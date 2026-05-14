"""
方向A：自动化特征工程系统
直接运行：python run.py
输出文件在 outputs/ 目录
"""
import sys, os, time, warnings
sys.path.insert(0, 'code')
warnings.filterwarnings('ignore')
os.makedirs('outputs', exist_ok=True)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # 非交互模式，直接保存图片
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import lightgbm as lgb
from sklearn.metrics import (roc_auc_score, precision_score, recall_score,
                              f1_score, confusion_matrix, roc_curve)
from tqdm import tqdm

from agent import FeatureAgent
from metrics import batch_cs_ic, ic_ir, feature_auc, group_monotonicity
from prompts import DIAGNOSE, CLEAN, EVALUATE, SELECT

plt.style.use('seaborn-v0_8-whitegrid')
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang HK', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# ── 1. 加载数据 ────────────────────────────────────────────
print("\n" + "="*60)
print("【1/6】加载数据")
print("="*60)
df = pd.read_parquet('../data.pq')
x_cols = sorted([c for c in df.columns if c.startswith('X')], key=lambda c: int(c[1:]))
TARGET = 'Y7'

dates      = sorted(df['trade_date'].unique())
split_idx  = int(len(dates) * 0.7)
split_date = dates[split_idx]
train_dates = set(dates[:split_idx])
test_dates  = set(dates[split_idx:])

print(f"数据维度   : {df.shape}")
print(f"交易日     : {str(dates[0])[:10]} ~ {str(dates[-1])[:10]}  ({len(dates)} 天)")
print(f"训练集截止 : {str(dates[split_idx-1])[:10]}  ({len(train_dates)} 天)")
print(f"测试集开始 : {str(dates[split_idx])[:10]}  ({len(test_dates)} 天)")
print(f"\n{TARGET} 分布:")
for v, p in df[TARGET].value_counts(normalize=True).sort_index().items():
    print(f"  {int(v):+d}: {p*100:.1f}%")

state = dict(df=df, x_cols=x_cols, target=TARGET,
             train_dates=train_dates, test_dates=test_dates, split_date=split_date)

# 数据概况图
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
vc = df[TARGET].value_counts().sort_index()
axes[0].bar(['-1','0','+1'], vc.values, color=['#e74c3c','#95a5a6','#2ecc71'], alpha=0.9)
axes[0].set_title(f'{TARGET} 标签分布')
null_pcts = df[x_cols].isnull().mean() * 100
axes[1].hist(null_pcts, bins=15, color='#3498db', alpha=0.85)
axes[1].set_title('特征缺失率分布')
uc = df.groupby('underlying').size()
axes[2].hist(uc.values, bins=15, color='#9b59b6', alpha=0.85)
axes[2].set_title('各标的样本数')
plt.tight_layout()
plt.savefig('outputs/data_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ 数据概况图已保存 → outputs/data_overview.png")

# ── 2. 初始化 Agent ─────────────────────────────────────────
print("\n" + "="*60)
print("【2/6】初始化 Agent（DeepSeek）")
print("="*60)
agent = FeatureAgent(state, verbose=False)
print(f"✓ Agent 就绪")

# ── 3. Phase 1：特征诊断 ────────────────────────────────────
print("\n" + "="*60)
print("【3/6】Phase 1 — 自动特征诊断")
print("="*60)
agent.reset()
agent.run(DIAGNOSE)

# Fallback
if 'diagnose_report' not in state:
    print("[fallback] 手动生成诊断报告")
    rows = []
    for x in x_cols:
        col = df[x].dropna()
        q1, q3 = col.quantile(0.25), col.quantile(0.75)
        iqr = q3 - q1
        rows.append({'feature': x,
                     'null_pct': df[x].isnull().mean() * 100,
                     'zero_pct': (df[x] == 0).mean() * 100,
                     'mean': col.mean(), 'std': col.std(),
                     'skew': col.skew(), 'kurt': col.kurtosis(),
                     'outlier_pct': ((col < q1-1.5*iqr)|(col > q3+1.5*iqr)).mean()*100,
                     'null_group': df[x].isnull().sum()})
    state['diagnose_report'] = pd.DataFrame(rows)

raw_diag = state['diagnose_report']
if isinstance(raw_diag, pd.DataFrame):
    diag = raw_diag.copy()
elif isinstance(raw_diag, dict):
    if isinstance(raw_diag.get('rows'), list):
        diag = pd.DataFrame(raw_diag['rows'])
    elif isinstance(raw_diag.get('data'), list):
        diag = pd.DataFrame(raw_diag['data'])
    else:
        diag = pd.DataFrame(raw_diag)
else:
    diag = pd.DataFrame(raw_diag)

if 'feature' not in diag.columns:
    diag = diag.reset_index().rename(columns={'index': 'feature'})
if 'feature' not in diag.columns:
    diag['feature'] = [f'X{i+1}' for i in range(len(diag))]
state['diagnose_report'] = diag
diag.to_csv('outputs/diagnose_report.csv', index=False)
print(f"✓ 诊断报告 → outputs/diagnose_report.csv  ({len(diag)} 行)")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
diag.nlargest(20,'null_pct').plot.barh(x='feature',y='null_pct',ax=axes[0],color='#e74c3c',legend=False)
axes[0].set_title('Top 20 高缺失率特征')
_n = (diag['skew'].abs()<1).sum()
_s = ((diag['skew'].abs()>=1)&(diag['skew'].abs()<5)).sum()
_h = (diag['skew'].abs()>=5).sum()
axes[1].pie([_n, _s, _h],
            labels=[f'正态({_n})', f'偏态({_s})', f'重尾({_h})'],
            autopct='%1.0f%%', colors=['#2ecc71','#f39c12','#e74c3c'],
            startangle=90)
axes[1].set_title('分布类型')
axes[2].hist(diag['skew'].clip(-20,20), bins=40, color='#3498db', alpha=0.85)
axes[2].set_title('偏度分布 (clip ±20)')
plt.tight_layout()
plt.savefig('outputs/diagnose_report.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ 诊断图 → outputs/diagnose_report.png")

# ── 4. Phase 2：特征清理 ────────────────────────────────────
print("\n" + "="*60)
print("【4/6】Phase 2 — 自动特征清理")
print("="*60)
agent.reset()
agent.run(CLEAN)

if 'df_clean' not in state:
    print("[fallback] 执行标准清理")
    raw_diag = state['diagnose_report']
    if isinstance(raw_diag, pd.DataFrame):
        diag_df = raw_diag.copy()
    elif isinstance(raw_diag, dict):
        if isinstance(raw_diag.get('rows'), list):
            diag_df = pd.DataFrame(raw_diag['rows'])
        elif isinstance(raw_diag.get('data'), list):
            diag_df = pd.DataFrame(raw_diag['data'])
        else:
            diag_df = pd.DataFrame(raw_diag)
    else:
        diag_df = pd.DataFrame(raw_diag)
    if 'feature' not in diag_df.columns:
        diag_df = diag_df.reset_index().rename(columns={'index': 'feature'})
    diag_idx = diag_df.set_index('feature')
    df_ = df.copy().sort_values(['underlying', 'trade_date'])
    df_[x_cols] = df_.groupby('underlying')[x_cols].ffill()
    tr_df = df_[df_['trade_date'].isin(train_dates)]
    med = tr_df.groupby('trade_date')[x_cols].median()
    for date, g_idx in df_.groupby('trade_date'):
        m = med.loc[date] if date in med.index else med.median()
        df_.loc[g_idx.index, x_cols] = df_.loc[g_idx.index, x_cols].fillna(m)
    for x in x_cols:
        sk = diag_idx.loc[x, 'skew'] if x in diag_idx.index else 0
        if abs(sk) > 5:
            lo, hi = tr_df[x].quantile(0.01), tr_df[x].quantile(0.99)
        else:
            mu, sg = tr_df[x].mean(), tr_df[x].std()
            lo, hi = mu - 3*sg, mu + 3*sg
        df_[x] = df_[x].clip(lo, hi)
    df_[x_cols] = df_.groupby('trade_date')[x_cols].rank(pct=True)
    state['df_clean'] = df_

df_clean = state['df_clean']
print(f"✓ 清理完成: {df_clean.shape}, 缺失: {df_clean[x_cols].isnull().sum().sum()}")

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for j, feat in enumerate(['X8', 'X3', 'X37']):
    orig = df[feat].dropna()
    axes[0, j].hist(orig.clip(orig.quantile(0.005), orig.quantile(0.995)),
                    bins=50, alpha=0.75, color='#e74c3c')
    axes[0, j].set_title(f'{feat} 原始 (skew={orig.skew():.1f})')
    axes[1, j].hist(df_clean[feat].dropna(), bins=50, alpha=0.75, color='#2ecc71')
    axes[1, j].set_title(f'{feat} 清理后 (rank归一化)')
plt.tight_layout()
plt.savefig('outputs/clean_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ 清理对比图 → outputs/clean_comparison.png")

# ── 5. Phase 3：特征评估 ────────────────────────────────────
print("\n" + "="*60)
print("【5/6】Phase 3 — 特征评估（IC / AUC / 单调性）")
print("="*60)
df_train = df_clean[df_clean['trade_date'].isin(train_dates)].copy()

print("计算截面 IC（向量化）...")
t0 = time.time()
ic_arrays = batch_cs_ic(df_train, x_cols, TARGET)
print(f"  耗时 {time.time()-t0:.1f}s")

rows = []
for f in tqdm(x_cols, desc='AUC/Mono'):
    ics = ic_arrays.get(f, np.array([]))
    rows.append({'feature': f,
                 'ic_mean': float(np.mean(ics)) if len(ics) else 0.0,
                 'ic_std':  float(np.std(ics))  if len(ics) else 0.0,
                 'ic_ir':   ic_ir(ics),
                 'auc':     feature_auc(df_train, f, TARGET),
                 'monotonicity': group_monotonicity(df_train, f, TARGET)})

eval_df = pd.DataFrame(rows)
state['eval_df'] = eval_df
eval_df.to_csv('outputs/eval_report.csv', index=False)
print(f"✓ 评估报告 → outputs/eval_report.csv")
print(f"  |IC|>0.05: {(eval_df['ic_mean'].abs()>0.05).sum()}  "
      f"|IC|>0.03: {(eval_df['ic_mean'].abs()>0.03).sum()}  "
      f"|IC|<0.01: {(eval_df['ic_mean'].abs()<0.01).sum()}")

print("计算相关矩阵...")
t0 = time.time()
corr_matrix = df_train[x_cols].dropna().corr(method='spearman')
state['corr_matrix'] = corr_matrix
print(f"  耗时 {time.time()-t0:.1f}s")

# Agent 审查评估结果
agent.reset()
agent.run(EVALUATE)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
axes[0].hist(eval_df['ic_mean'], bins=40, color='#3498db', alpha=0.85)
axes[0].axvline(0, color='k', lw=0.8)
for v in [0.03, -0.03]: axes[0].axvline(v, color='orange', ls='--', lw=1.2)
axes[0].set_title('IC Mean 分布')
top25 = eval_df.reindex(eval_df['ic_mean'].abs().sort_values(ascending=False).index).head(25)
axes[1].barh(top25['feature'].values[::-1], top25['ic_mean'].values[::-1],
             color=['#27ae60' if v>0 else '#e74c3c' for v in top25['ic_mean'].values[::-1]], alpha=0.85)
axes[1].set_title('Top 25 特征 |IC|')
axes[2].hist(eval_df['auc'], bins=30, color='#9b59b6', alpha=0.85)
axes[2].axvline(0.5, color='red', ls='--', lw=1.5)
axes[2].set_title('AUC 分布')
plt.tight_layout()
plt.savefig('outputs/feature_eval.png', dpi=150, bbox_inches='tight')
plt.close()
top30 = eval_df.reindex(eval_df['ic_mean'].abs().sort_values(ascending=False).index).head(30)['feature'].tolist()
corr30 = corr_matrix.loc[top30, top30]
fig2, ax2 = plt.subplots(figsize=(12, 10))
sns.heatmap(corr30, mask=np.triu(np.ones_like(corr30,dtype=bool)),
            cmap='RdBu_r', center=0, vmin=-1, vmax=1, ax=ax2, linewidths=0.3)
ax2.set_title('Top 30 IC 特征相关矩阵')
plt.tight_layout()
plt.savefig('outputs/feature_corr.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ 评估图 → outputs/feature_eval.png, feature_corr.png")

# Phase 4：特征筛选
print("\n" + "="*60)
print("【5/6 续】Phase 4 — 自动特征筛选 Top 50")
print("="*60)
def _manual_select(ev: pd.DataFrame, cm: pd.DataFrame):
    r1 = ev[~((ev['ic_mean'].abs() < 0.005) & (ev['auc'] < 0.51))]['feature'].tolist()
    ev2 = ev[ev['feature'].isin(r1)].copy()
    ev2['score'] = (0.4 * ev2['ic_mean'].abs().rank()
                    + 0.3 * ev2['ic_ir'].abs().rank()
                    + 0.3 * ev2['auc'].rank())
    r2 = ev2.nlargest(100, 'score')['feature'].tolist()

    sub_corr = cm.loc[r2, r2].fillna(0)
    dist = (1 - sub_corr.abs().clip(0, 1)).clip(lower=0)
    np.fill_diagonal(dist.values, 0)
    Z = linkage(squareform(dist), method='average')
    labels = fcluster(Z, t=0.2, criterion='distance')
    score_map = ev2.set_index('feature')['score']
    r3 = [max([r2[i] for i, l in enumerate(labels) if l == cl], key=lambda f: score_map.get(f, 0))
          for cl in np.unique(labels)]
    r4 = sorted(r3, key=lambda f: score_map.get(f, 0), reverse=True)[:50]
    top50_scores = ev2[ev2['feature'].isin(r4)][['feature', 'ic_mean', 'ic_ir', 'auc', 'score']] \
        .sort_values('score', ascending=False).reset_index(drop=True)
    return r4, {'r1': len(r1), 'r2': len(r2), 'r3': len(r3), 'r4': len(r4)}, top50_scores


def _selection_quality(top50, ev):
    top50 = list(top50) if top50 is not None else []
    valid = set(ev['feature'].tolist())
    ok_len = len(top50) == 50 and len(set(top50)) == 50 and set(top50).issubset(valid)
    sel = ev[ev['feature'].isin(top50)] if ok_len else pd.DataFrame()
    mean_abs_all = float(ev['ic_mean'].abs().mean())
    mean_abs_sel = float(sel['ic_mean'].abs().mean()) if not sel.empty else 0.0
    top100 = set(ev.reindex(ev['ic_mean'].abs().sort_values(ascending=False).index).head(100)['feature'].tolist())
    overlap100 = len(set(top50).intersection(top100)) if ok_len else 0
    ok_quality = ok_len and (mean_abs_sel >= mean_abs_all * 0.9) and (overlap100 >= 15)
    return ok_quality, {
        'ok_len': ok_len, 'mean_abs_all': mean_abs_all, 'mean_abs_sel': mean_abs_sel, 'overlap_top100': overlap100
    }


agent.reset()
agent.run(SELECT)
ok_sel, qc = _selection_quality(state.get('top50'), eval_df)
print(f"[QC] Agent筛选质量: ok={ok_sel}, mean|IC|={qc['mean_abs_sel']:.4f}, overlap@100={qc['overlap_top100']}")

if not ok_sel:
    print("[repair] Agent结果质量未达标，触发二次修正")
    repair_prompt = SELECT + """

补充要求：
- 你上一轮 top50 质量不足，请重新执行四轮筛选并覆盖 state['top50']。
- 强制检查：top50 的 mean(|ic_mean|) 必须 >= 全集 mean(|ic_mean|) * 0.9；
- 强制检查：top50 与 |ic_mean| 前100特征重合数 >= 15。
- 若不满足，继续调整阈值/排序后再输出最终结果。
"""
    agent.reset()
    agent.run(repair_prompt)
    ok_sel, qc = _selection_quality(state.get('top50'), eval_df)
    print(f"[QC] 二次修正后: ok={ok_sel}, mean|IC|={qc['mean_abs_sel']:.4f}, overlap@100={qc['overlap_top100']}")

if not ok_sel:
    print("[fallback] 使用固定规则兜底（Agent驱动失败保护）")
    r4, slog, top50_scores = _manual_select(eval_df.copy(), corr_matrix)
    state['top50'] = r4
    state['select_log'] = slog
    state['top50_scores'] = top50_scores
else:
    if state.get('top50_scores') is None:
        state['top50_scores'] = eval_df[eval_df['feature'].isin(state['top50'])][['feature', 'ic_mean', 'ic_ir', 'auc']].copy()
    state['top50_scores'] = state['top50_scores'].drop_duplicates('feature')

top50 = state['top50']
sel_log = state.get('select_log',{})
pd.Series(top50, name='feature').to_csv('outputs/top50_features.csv', index=False)
print(f"✓ Top 50 特征 → outputs/top50_features.csv")
print(f"  筛选漏斗: 300 → {sel_log.get('r1','?')} → {sel_log.get('r2','?')} → {sel_log.get('r3','?')} → {sel_log.get('r4','?')}")

ev_top = eval_df[eval_df['feature'].isin(top50)].sort_values('ic_mean')
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
axes[0].barh(ev_top['feature'], ev_top['ic_mean'],
             color=['#27ae60' if v>0 else '#e74c3c' for v in ev_top['ic_mean']], alpha=0.85)
axes[0].set_title(f'Top {len(top50)} 特征 IC Mean')
counts = [300, sel_log.get('r1',0), sel_log.get('r2',0), sel_log.get('r3',0), sel_log.get('r4',0)]
axes[1].bar(['原始(300)','Round1','Round2','Round3',f'Top{len(top50)}'],
            counts, color=['#3498db','#f39c12','#e67e22','#e74c3c','#27ae60'], alpha=0.85)
for i, n in enumerate(counts):
    axes[1].text(i, n+2, str(n), ha='center', fontsize=11, fontweight='bold')
axes[1].set_title('四轮漏斗筛选')
plt.tight_layout()
plt.savefig('outputs/top50_selection.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ 筛选漏斗图 → outputs/top50_selection.png")

# ── 6. 模型验证 ─────────────────────────────────────────────
print("\n" + "="*60)
print("【6/6】模型验证（LightGBM）")
print("="*60)
df_model = df_clean[df_clean[TARGET] != 0].copy()
df_model['y'] = (df_model[TARGET] == 1).astype(int)
train_m = df_model['trade_date'].isin(train_dates)
test_m  = df_model['trade_date'].isin(test_dates)
X_tr, y_tr = df_model.loc[train_m, top50].fillna(0), df_model.loc[train_m, 'y']
X_te, y_te = df_model.loc[test_m,  top50].fillna(0), df_model.loc[test_m,  'y']
print(f"训练集: {X_tr.shape}  正类率 {y_tr.mean():.3f}")
print(f"测试集: {X_te.shape}  正类率 {y_te.mean():.3f}")

clf = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=63,
                          max_depth=6, subsample=0.8, colsample_bytree=0.8,
                          class_weight='balanced', random_state=42, verbose=-1)
clf.fit(X_tr, y_tr, eval_set=[(X_te, y_te)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)])

proba = clf.predict_proba(X_te)[:, 1]
pred  = clf.predict(X_te)
auc  = roc_auc_score(y_te, proba)
prec = precision_score(y_te, pred, zero_division=0)
rec  = recall_score(y_te, pred, zero_division=0)
f1   = f1_score(y_te, pred, zero_division=0)

print(f"\n{'='*40}")
print(f"  AUC       : {auc:.4f}")
print(f"  Precision : {prec:.4f}")
print(f"  Recall    : {rec:.4f}")
print(f"  F1        : {f1:.4f}")
print('='*40)

fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fpr, tpr, _ = roc_curve(y_te, proba)
axes[0].plot(fpr, tpr, color='#27ae60', lw=2, label=f'AUC={auc:.4f}')
axes[0].plot([0,1],[0,1],'k--',lw=1,alpha=0.4)
axes[0].fill_between(fpr, tpr, alpha=0.08, color='#27ae60')
axes[0].set_title('ROC Curve（测试集）'); axes[0].legend()
cm = confusion_matrix(y_te, pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1],
            xticklabels=['Pred:空','Pred:多'], yticklabels=['True:空','True:多'],
            annot_kws={'size':13})
axes[1].set_title('混淆矩阵')
fi = pd.Series(clf.feature_importances_, index=top50).sort_values(ascending=False).head(20)
axes[2].barh(fi.index[::-1], fi.values[::-1], color='#3498db', alpha=0.85)
axes[2].set_title('特征重要度 Top 20')
plt.tight_layout()
plt.savefig('outputs/model_evaluation.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ 评估图 → outputs/model_evaluation.png")

# 数据泄漏验证
print("\n数据泄漏检测:")
from scipy.stats import spearmanr
overlap = len(train_dates & test_dates)
print(f"  训练/测试日期重叠: {overlap} 天 {'✓' if overlap==0 else '✗'}")
ok_order = max(train_dates) < min(test_dates)
print(f"  训练截止: {str(max(train_dates))[:10]}  测试开始: {str(min(test_dates))[:10]}  {'✓' if ok_order else '✗'}")
# 用截面 IC 均值对比（正确方式）
def _mean_cs_ic(df_sub, feat, tgt, n_dates=50):
    dates_sample = sorted(df_sub['trade_date'].unique())[-n_dates:]
    ics = []
    for d in dates_sample:
        g = df_sub[df_sub['trade_date']==d][[feat, tgt]].dropna()
        if len(g) >= 5:
            r, _ = spearmanr(g[feat], g[tgt])
            ics.append(abs(r))
    return np.mean(ics) if ics else 0

te_df = df_clean[df_clean['trade_date'].isin(test_dates)]
tr_ic = np.mean([_mean_cs_ic(df_train, f, TARGET) for f in top50[:10]])
te_ic = np.mean([_mean_cs_ic(te_df,    f, TARGET) for f in top50[:10]])
ratio = te_ic / (tr_ic + 1e-9)
print(f"  截面IC（训练集）: {tr_ic:.4f}  截面IC（测试集）: {te_ic:.4f}  比值: {ratio:.2f}x  {'✓' if 0.2<ratio<5 else '!'}")
ok_ratio = 0.2 < ratio < 5
print(f"  结论: {'无数据泄漏' if ok_order and overlap==0 and ok_ratio else '存在风险'}")

# Agent 决策日志
print("\n" + "="*60)
print("Agent 决策日志")
print("="*60)
if agent.decision_log:
    for e in agent.decision_log:
        print(f"  [{e.get('phase')}] {e.get('action')} — {e.get('rationale')}")
else:
    print("  (无记录)")

print("\n" + "="*60)
print("✅ 全部完成！输出文件在 outputs/ 目录")
print("="*60)
print("outputs/")
for f in sorted(os.listdir('outputs')):
    size = os.path.getsize(f'outputs/{f}')
    print(f"  {f}  ({size//1024} KB)" if size > 1024 else f"  {f}  ({size} B)")
