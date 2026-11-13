# -*- coding: utf-8 -*-
"""
独立研究③：轮廓系数 silhouette_score 是什么——用代码算一次
定义（对单个点 i）：
  a(i) = i 到**同簇**其他点的平均距离（内聚度，越小越好）
  b(i) = i 到**最近的其他簇**各点的平均距离的最小值（分离度，越大越好）
  s(i) = (b(i) - a(i)) / max(a(i), b(i))，取值 [-1, 1]
全局轮廓系数 = 所有点 s(i) 的平均值。
直觉：s→1 表示"离自家簇很近、离别家簇很远"（分得好）；
     s→0 表示点在两簇边界上；s<0 表示点多半被分错了簇。
产出：outputs/23_silhouette.png
"""
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples

matplotlib.rcParams["font.sans-serif"] = ["Songti SC", "STHeiti", "PingFang SC", "Microsoft YaHei", "SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

FEATS = ["income", "balance", "age"]
df = pd.read_csv("data/mall_customers_sim.csv")
X = StandardScaler().fit_transform(df[FEATS])
labels = KMeans(n_clusters=4, random_state=42, n_init=10).fit_predict(X)

# ---------- 第 1 步：手写单点 a/b/s（以第 0 个点为例） ----------
i = 0
d_same = [np.linalg.norm(X[i] - X[j]) for j in range(len(X)) if labels[j] == labels[i] and j != i]
a_i = np.mean(d_same)
b_i = min(np.mean([np.linalg.norm(X[i] - X[j]) for j in range(len(X)) if labels[j] == c])
          for c in set(labels) if c != labels[i])
s_i = (b_i - a_i) / max(a_i, b_i)
print(f"手算第 0 个点：a={a_i:.4f}（同簇平均距离） b={b_i:.4f}（最近异簇平均距离）")
print(f"  s=(b-a)/max(a,b)=({b_i:.4f}-{a_i:.4f})/{max(a_i,b_i):.4f}={s_i:.4f}")

# ---------- 第 2 步：sklearn 全局轮廓系数 ----------
global_s = silhouette_score(X, labels)
print(f"\nsklearn 全局轮廓系数：{global_s:.4f}")

# ---------- 第 3 步：每个点的轮廓系数分布图（按簇排序） ----------
samples = silhouette_samples(X, labels)
order = np.argsort(labels, kind="stable")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
colors = ["#DD8452", "#4C72B0", "#55A868", "#C44E52"]
y_lower = 0
for c in range(4):
    sc = np.sort(samples[labels == c])
    y_upper = y_lower + len(sc)
    axes[0].fill_betweenx(np.arange(y_lower, y_upper), 0, sc,
                          color=colors[c], alpha=0.8)
    axes[0].text(-0.05, (y_lower + y_upper) / 2, f"簇 {c}", ha="right", fontsize=11)
    y_lower = y_upper
axes[0].axvline(global_s, color="red", linestyle="--", label=f"全局平均 {global_s:.4f}")
axes[0].set_title("各点轮廓系数分布（按簇分层）", fontsize=13)
axes[0].set_xlabel("轮廓系数 s(i)"); axes[0].set_ylabel("样本（按簇排列）")
axes[0].legend(); axes[0].grid(alpha=0.3, axis="x")

axes[1].scatter(X[:, 0], X[:, 1], s=30, c=labels,
                cmap=matplotlib.colors.ListedColormap(colors), alpha=0.75)
axes[1].scatter(X[0, 0], X[0, 1], s=350, marker="o", facecolors="none",
                edgecolors="red", linewidths=2.5, label=f"手算示例点（s={s_i:.3f}）")
axes[1].set_title(f"聚类着色（红圈为手算示例点）", fontsize=13)
axes[1].set_xlabel("income（标准化后）"); axes[1].set_ylabel("balance（标准化后）")
axes[1].legend(); axes[1].grid(alpha=0.3)
fig.suptitle("独立研究③：轮廓系数——内聚度 a 与分离度 b 之比，取值 [-1, 1]", fontsize=15)
plt.tight_layout()
plt.savefig("outputs/23_silhouette.png", dpi=150)
print("已保存 outputs/23_silhouette.png")
