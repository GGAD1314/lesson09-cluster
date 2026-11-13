# -*- coding: utf-8 -*-
"""
自主实践①：换特征集——加入 age 重新聚类，对比画像变化
特征：income / balance / age（三特征，标准化后）
产出：
  outputs/11_age_elbow.png     肘部法+轮廓系数双证据图
  outputs/11_age_profile.png   四簇画像分组柱状图
  outputs/11_age_scatter.png   income-balance 平面按四簇着色散点图
  generated/age_report.md      新画像报告（四簇中文业务命名 + 与两特征版对比）
"""
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

matplotlib.rcParams["font.sans-serif"] = ["Songti SC", "STHeiti", "PingFang SC", "Microsoft YaHei", "SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

FEATS = ["income", "balance", "age"]

df = pd.read_csv("data/mall_customers_sim.csv")
X = StandardScaler().fit_transform(df[FEATS])

# ---------- 第 1 步：肘部法 + 轮廓系数双证据选 k ----------
ks = range(1, 10)
inertias = []
for k in ks:
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
    inertias.append(km.inertia_)

ks2 = range(2, 10)
sils = []
for k in ks2:
    labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
    sils.append(silhouette_score(X, labels))

print("肘部法 inertia 表：")
for k, v in zip(ks, inertias):
    print(f"  k={k}: inertia={v:.2f}")
print("轮廓系数表：")
for k, v in zip(ks2, sils):
    print(f"  k={k}: silhouette={v:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 7))
axes[0].plot(list(ks), inertias, "o-", color="#4C72B0", linewidth=2)
axes[0].axvline(x=4, color="red", linestyle="--", alpha=0.6, label="业务取 k=4")
axes[0].set_title("三特征（收入/余额/年龄）肘部法", fontsize=13)
axes[0].set_xlabel("簇数 k"); axes[0].set_ylabel("簇内平方和 inertia")
axes[0].set_xticks(list(ks)); axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].plot(list(ks2), sils, "s-", color="#55A868", linewidth=2)
axes[1].axvline(x=4, color="red", linestyle="--", alpha=0.6, label="业务取 k=4")
axes[1].set_title("三特征轮廓系数", fontsize=13)
axes[1].set_xlabel("簇数 k"); axes[1].set_ylabel("轮廓系数 silhouette")
axes[1].set_xticks(list(ks2)); axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/11_age_elbow.png", dpi=150)
print("\n已保存 outputs/11_age_elbow.png")

# ---------- 第 2 步：k=4 聚类 + 画像 ----------
K = 4
km = KMeans(n_clusters=K, random_state=42, n_init=10)
df["cluster"] = km.fit_predict(X)

summary = df.groupby("cluster").agg(
    人数=("customer_id", "count"),
    平均年龄=("age", "mean"),
    平均收入万=("income", "mean"),
    平均余额万=("balance", "mean"),
    平均持有产品数=("product_count", "mean"),
).round(2)
print("\n四簇画像：\n", summary.to_string())

# 按平均收入排序起业务名（低→高）
order = summary["平均收入万"].sort_values().index.tolist()
biz_names = {
    order[0]: "低收低余的年轻客群",
    order[1]: "中收中余的新锐白领客群",   # 中产里较年轻的一半
    order[2]: "中收中余的中年主力客群",   # 中产里年长的一半
    order[3]: "高收高余的富裕客群",
}

# ---------- 第 3 步：画像分组柱状图 ----------
feats_cn = {"平均年龄": "年龄（岁）", "平均收入万": "收入（万元）",
            "平均余额万": "余额（万元）", "平均持有产品数": "持有产品数（个）"}
fig, axes = plt.subplots(1, 4, figsize=(16, 5))
colors = ["#DD8452", "#4C72B0", "#55A868", "#C44E52"]
for ax, feat in zip(axes, feats_cn):
    for i, c in enumerate(order):
        ax.bar(i, summary.loc[c, feat], color=colors[i], alpha=0.85)
        ax.text(i, summary.loc[c, feat], f"{summary.loc[c, feat]:.1f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_title(feats_cn[feat], fontsize=12)
    ax.set_xticks(range(K))
    ax.set_xticklabels([f"簇{c}\n{summary.loc[c,'人数']}人" for c in order], fontsize=9)
    ax.grid(alpha=0.3, axis="y")
fig.suptitle("加入 age 后的四簇画像（k=4，标准化后）", fontsize=15)
plt.tight_layout()
plt.savefig("outputs/11_age_profile.png", dpi=150)
print("已保存 outputs/11_age_profile.png")

# ---------- 第 4 步：income-balance 平面按四簇着色 ----------
fig, ax = plt.subplots(figsize=(10, 7))
for i, c in enumerate(sorted(df["cluster"].unique())):
    sub = df[df["cluster"] == c]
    ax.scatter(sub["income"], sub["balance"], s=40, alpha=0.7,
               color=colors[list(order).index(c)],
               label=f"{biz_names[c]}（{len(sub)} 人）")
centers_raw = StandardScaler().fit(df[FEATS]).inverse_transform(km.cluster_centers_)
ax.scatter(centers_raw[:, 0], centers_raw[:, 1], marker="*", s=600,
           c="red", edgecolors="black", linewidths=1.2, zorder=5, label="簇中心")
ax.set_title("加入 age 后的聚类结果（income-balance 平面投影，k=4）", fontsize=15)
ax.set_xlabel("年收入 income（万元）", fontsize=12)
ax.set_ylabel("账户平均余额 balance（万元）", fontsize=12)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/11_age_scatter.png", dpi=150)
print("已保存 outputs/11_age_scatter.png")

# ---------- 第 5 步：对比报告 ----------
lines = ["# 自主实践①：加入 age 重新画像（k=4）", "",
         "## 选 k 依据",
         f"肘部法：inertia 从 k=3 的 {inertias[2]:.2f} 降到 k=4 的 {inertias[3]:.2f}"
         f"（再往后 k=5 仅降到 {inertias[4]:.2f}，下降明显变缓）；"
         f"轮廓系数 k=3 为 {sils[1]:.4f}、k=4 为 {sils[2]:.4f}。"
         "数学证据在 3~4 之间偏 3，但 k=3 时中产簇内部明显存在两个年龄段"
         "（30 岁 vs 43 岁），业务上拆开打法更精细——**选 k 是业务决策，取 k=4**。", "",
         "## 四簇画像与业务命名", "", summary.to_string(), ""]
for c in order:
    row = summary.loc[c]
    lines.append(f"- **簇 {c} · {biz_names[c]}**：{int(row['人数'])} 人"
                 f"（{row['人数']/300*100:.1f}%），平均年龄 {row['平均年龄']} 岁、"
                 f"平均收入 {row['平均收入万']} 万元、平均余额 {row['平均余额万']} 万元、"
                 f"平均持有 {row['平均持有产品数']} 个产品。")
lines += ["", "## 与两特征版（126/124/50）的画像变化", "",
          "两特征版中中产是一整块（124 人）；加入 age 后中产按年龄**一分为二**："
          "约 30 岁的新锐白领与约 43 岁的中年主力，两人群收入余额几乎相同、"
          "生命阶段不同——前者适合消费信贷与成长型理财，后者适合教育金、"
          "保险与家庭资产配置。富裕（50 人）与年轻（121 人）两群基本不变。", "",
          "## 一句话结论",
          "**特征决定分群含义**：加 age 前分的是「价值层级」，加 age 后分的是"
          "「价值层级 × 生命阶段」，客群从 3 层变 4 层，运营打法可以更细。"]
with open("generated/age_report.md", "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print("已保存 generated/age_report.md")

# 终端验收三问
print("\n【验收】四簇人数：", dict(summary["人数"]))
print("【验收】四中文命名：", [biz_names[c] for c in order])
