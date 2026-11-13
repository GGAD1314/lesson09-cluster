# -*- coding: utf-8 -*-
"""
独立研究②：k-均值对"环形分布"数据会失败——用玩具数据验证，DBSCAN 一句话对比
生成两个同心圆环（内环 + 外环），人眼一看就是两群：
  - KMeans 按欧氏距离把空间"切两半"，得到两个半圆——**失败**（它假设簇是球形）
  - DBSCAN 按密度连通，把同一环上的点连成一簇——**成功**
一句话对比：k-均值分的是"到中心谁近"（球形假设），DBSCAN 分的是"密度连成片"（任意形状）。
产出：outputs/22_ring_dbscan.png
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler

matplotlib.rcParams["font.sans-serif"] = ["Songti SC", "STHeiti", "PingFang SC", "Microsoft YaHei", "SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

X, y_true = make_circles(n_samples=800, factor=0.35, noise=0.06, random_state=42)
X = StandardScaler().fit_transform(X)

# ---------- KMeans：失败 ----------
km = KMeans(n_clusters=2, random_state=42, n_init=10)
labels_km = km.fit_predict(X)

# ---------- DBSCAN：成功 ----------
db = DBSCAN(eps=0.25, min_samples=5)
labels_db = db.fit_predict(X)
n_noise = (labels_db == -1).sum()
print(f"KMeans 两簇人数：{np.bincount(labels_km)}（各含内环+外环的一半——分错了）")
print(f"DBSCAN 簇数：{len(set(labels_db)) - (1 if -1 in labels_db else 0)}，"
      f"各簇人数：{np.bincount(labels_db[labels_db >= 0])}，噪声点：{n_noise}")
# 验证 DBSCAN 分对了：每个簇的真实标签应唯一
for c in sorted(set(labels_db) - {-1}):
    true_labels = set(y_true[labels_db == c])
    print(f"  DBSCAN 簇 {c} 只包含真实环 {true_labels} —— {'✓ 正确' if len(true_labels)==1 else '✗ 混环'}")

# ---------- 可视化 ----------
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].scatter(X[:, 0], X[:, 1], s=12, c=y_true, cmap="coolwarm", alpha=0.7)
axes[0].set_title("真实结构：两个同心圆环（玩具数据）", fontsize=13)
axes[1].scatter(X[:, 0], X[:, 1], s=12, c=labels_km, cmap="coolwarm", alpha=0.7)
axes[1].scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
                marker="*", s=500, c="red", edgecolors="black", zorder=5)
axes[1].set_title("KMeans（k=2）：切成两个半圆——失败", fontsize=13)
noise = labels_db == -1
for c, color in zip(sorted(set(labels_db) - {-1}), ["#DD8452", "#4C72B0", "#55A868"]):
    sub = X[labels_db == c]
    axes[2].scatter(sub[:, 0], sub[:, 1], s=12, color=color, alpha=0.7, label=f"簇 {c}")
axes[2].scatter(X[noise, 0], X[noise, 1], s=12, c="lightgray", label=f"噪声（{n_noise} 点）")
axes[2].set_title("DBSCAN：沿密度连成环——成功", fontsize=13)
for ax in axes:
    ax.set_xlabel("特征 1（标准化后）"); ax.set_ylabel("特征 2（标准化后）")
    ax.grid(alpha=0.3)
axes[2].legend(fontsize=10)
fig.suptitle("独立研究②：环形数据上 k-均值失败 vs DBSCAN 成功", fontsize=15)
plt.tight_layout()
plt.savefig("outputs/22_ring_dbscan.png", dpi=150)
print("已保存 outputs/22_ring_dbscan.png")
