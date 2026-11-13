# -*- coding: utf-8 -*-
"""
真实数据重跑：data/mall_customers_yuan.csv（balance 以元计）
流程与上一轮完全一致：初检 → KMeans(k=3, random_state=42, income+balance) → 聚类图 → 报告
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.cluster import KMeans

matplotlib.rcParams["font.sans-serif"] = ["Songti SC", "STHeiti", "PingFang SC", "Microsoft YaHei", "SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

# ---- 1. 数据初检 ----
df = pd.read_csv("data/mall_customers_yuan.csv")
print(f"行列数: {df.shape}")
print("各列缺失值:\n", df.isnull().sum().to_string())
print("数值列统计:\n", df[["age", "income", "balance", "product_count"]].describe().round(2).to_string())

# ---- 2. 重跑聚类（k=3、random_state=42，其他不动——故意未标准化）----
X = df[["income", "balance"]]
km = KMeans(n_clusters=3, random_state=42)
df["cluster"] = km.fit_predict(X)

counts = df["cluster"].value_counts().sort_index()
print("\n各簇人数:")
for c, n in counts.items():
    stat = df[df["cluster"] == c][["income", "balance"]].mean()
    print(f"  簇 {c}: {n} 人 | 平均收入 {stat['income']:.1f} 万 | 平均余额 {stat['balance']:,.0f} 元")

# ---- 3. 重画聚类图 ----
fig, ax = plt.subplots(figsize=(10, 7), dpi=150)
colors = ["#4C72B0", "#DD8452", "#55A868"]
for c in sorted(df["cluster"].unique()):
    sub = df[df["cluster"] == c]
    ax.scatter(sub["income"], sub["balance"], s=28, c=colors[c], alpha=0.75,
               label=f"簇 {c}（{len(sub)} 人）")
ax.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
           marker="*", s=520, c="#B03A2E", edgecolors="white", linewidths=1.2,
           zorder=5, label="簇中心")
ax.set_xlabel("年收入（万元）")
ax.set_ylabel("账户平均余额（元）")
ax.set_title("真实数据重跑：收入-余额聚类（k=3，random_state=42，余额单位：元）")
ax.legend()
ax.yaxis.set_major_formatter(lambda v, _: f"{v/10000:.0f}万")
fig.tight_layout()
fig.savefig("outputs/02_yuan_fail.png")
print("\n图已保存: outputs/02_yuan_fail.png")

# ---- 4. 更新分群报告（按元计新数据重新统计，业务名按平均收入排序重新起）----
order = df.groupby("cluster")["income"].mean().sort_values().index.tolist()
biz = {order[0]: "低收低余的年轻客群", order[1]: "中收中余的主力客群", order[2]: "高收高余的富裕客群"}
prev = {"低收低余的年轻客群": 125, "中收中余的主力客群": 125, "高收高余的富裕客群": 50}

report_lines = ["# 分群报告（真实导出数据：余额以元计，k=3 重跑）", "",
                "数据源：data/mall_customers_yuan.csv（300 行 × 7 列，无缺失；balance 为系统原始单位·元）。"
                "代码与参数同上一轮（特征 income/balance、k=3、random_state=42），仅数据源更换。", "",
                "| 簇 | 业务名 | 人数 | 占比 | 平均年龄 | 平均收入(万) | 平均余额(元) | 平均产品数 |",
                "|---|---|---|---|---|---|---|---|"]
for c in order:
    g = df[df["cluster"] == c]
    report_lines.append(
        f"| {c} | {biz[c]} | {len(g)} | {len(g)/len(df)*100:.1f}% | {g['age'].mean():.0f} | "
        f"{g['income'].mean():.1f} | {g['balance'].mean():,.0f} | {g['product_count'].mean():.1f} |")
report_lines += ["", "## 与上一轮人数对照（上一轮：125 / 125 / 50）", "",
                 "| 业务名 | 上一轮 | 本轮 | 变化 |",
                 "|---|---|---|---|"]
for c in order:
    n = int((df["cluster"] == c).sum())
    report_lines.append(f"| {biz[c]} | {prev[biz[c]]} | {n} | {n - prev[biz[c]]:+d} |")
with open("generated/mall_report.md", "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines) + "\n")
print("报告已更新: generated/mall_report.md")
