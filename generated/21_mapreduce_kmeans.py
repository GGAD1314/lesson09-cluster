# -*- coding: utf-8 -*-
"""
独立研究①：MapReduce 并行 k-均值（教材 9.3）——单机模拟 map/reduce 两个阶段
解决什么问题：数据量大到单机内存/算力装不下时，把 k-均值的两大步拆到多台机器并行——
  map 阶段：每个数据点独立计算"离哪个质心最近"→ 发射 (簇编号, 点坐标)，点与点之间互不依赖，天然可并行；
  reduce 阶段：每个 reducer 负责一个簇，把收到的点求均值得到新质心，簇与簇之间互不依赖，也可并行。
只有质心这一极小的中间结果需要在每轮迭代开始时广播给所有 mapper。
本脚本用 Python 函数模拟这两个阶段（数据小，单机串行模拟分布式逻辑），
并与 sklearn 的 KMeans 对比，验证结果一致。
产出：outputs/21_mapreduce_vs_sklearn.png
"""
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

matplotlib.rcParams["font.sans-serif"] = ["Songti SC", "STHeiti", "PingFang SC", "Microsoft YaHei", "SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

FEATS = ["income", "balance", "age"]
K = 4
MAX_ITER = 50

df = pd.read_csv("data/mall_customers_sim.csv")
X = StandardScaler().fit_transform(df[FEATS])
rng = np.random.default_rng(42)


# ---------- k-means++ 初始化（与 sklearn 默认一致） ----------
def kmeanspp_init(X, K, rng):
    centers = [X[rng.integers(len(X))]]
    for _ in range(K - 1):
        d2 = np.min(((X[:, None, :] - np.array(centers)[None, :, :]) ** 2).sum(axis=2), axis=1)
        centers.append(X[rng.choice(len(X), p=d2 / d2.sum())])
    return np.array(centers)


# ---------- map 阶段：每个点独立找最近质心（可分片到多台机器） ----------
def mapper(points, centers):
    """输入一片数据点 + 当前质心，输出 [(簇编号, 点), ...]"""
    out = []
    for p in points:
        c = int(np.argmin(((centers - p) ** 2).sum(axis=1)))
        out.append((c, p))
    return out


# ---------- reduce 阶段：每个簇独立求均值（一个 reducer 负责一个簇） ----------
def reducer(pairs, K):
    """按簇编号汇总，输出新质心"""
    sums = np.zeros((K, X.shape[1]))
    counts = np.zeros(K)
    for c, p in pairs:
        sums[c] += p
        counts[c] += 1
    return sums / counts[:, None]


# ---------- 主循环：map → reduce → 广播新质心，直到收敛 ----------
# 与 sklearn 的 n_init=10 对应：多次随机初始化重启，取 inertia 最优的一轮
best_centers, best_pairs, best_inertia = None, None, np.inf
for restart in range(10):
    centers = kmeanspp_init(X, K, rng)
    for it in range(MAX_ITER):
        pairs = mapper(X, centers)          # ① map：全体点分配（分布式并行）
        new_centers = reducer(pairs, K)     # ② reduce：各簇求均值（分布式并行）
        shift = np.abs(new_centers - centers).max()
        if shift < 1e-8:
            break
        centers = new_centers
    inertia = sum(((p - centers[c]) ** 2).sum() for c, p in pairs)
    if inertia < best_inertia:
        best_centers, best_pairs, best_inertia = centers, pairs, inertia
centers, pairs, inertia = best_centers, best_pairs, best_inertia
print(f"10 次重启取最优：inertia={inertia:.4f}")

labels_mr = np.array([c for c, _ in pairs])

# ---------- 与 sklearn 对比 ----------
km = KMeans(n_clusters=K, random_state=42, n_init=10).fit(X)
# 对齐簇编号（按收入排序）后比较
def relabel_by_income(labels, centers):
    order = np.argsort(centers[:, 0])
    remap = {old: new for new, old in enumerate(order)}
    return np.array([remap[l] for l in labels]), remap

labels_mr_al, _ = relabel_by_income(labels_mr, centers)
labels_sk_al, _ = relabel_by_income(km.labels_, km.cluster_centers_)
agree = (labels_mr_al == labels_sk_al).mean()
print(f"\nMapReduce 模拟 与 sklearn KMeans 标签一致率：{agree*100:.1f}%")
print(f"inertia：模拟 {inertia:.4f} vs sklearn {km.inertia_:.4f}")

# ---------- 可视化 ----------
fig, axes = plt.subplots(1, 2, figsize=(14, 7))
colors = ["#DD8452", "#4C72B0", "#55A868", "#C44E52"]
for ax, lab, title in [(axes[0], labels_mr_al, "MapReduce 模拟版（map 分配 + reduce 更新）"),
                        (axes[1], labels_sk_al, "sklearn KMeans（对照）")]:
    for c in range(K):
        sub = X[lab == c]
        ax.scatter(sub[:, 0], sub[:, 1], s=30, alpha=0.75, color=colors[c], label=f"簇 {c}（{len(sub)} 人）")
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("income（标准化后）"); ax.set_ylabel("balance（标准化后）")
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
fig.suptitle(f"独立研究①：MapReduce 并行 k-均值模拟——与 sklearn 结果一致率 {agree*100:.1f}%", fontsize=15)
plt.tight_layout()
plt.savefig("outputs/21_mapreduce_vs_sklearn.png", dpi=150)
print("已保存 outputs/21_mapreduce_vs_sklearn.png")
