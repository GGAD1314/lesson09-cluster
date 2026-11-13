# -*- coding: utf-8 -*-
"""
自主实践③：scatter 二维投影——任选两特征着色 by cluster
三张散点图：income×balance / income×age / balance×age（颜色 = 四簇编号）
产出：outputs/13_scatter_pairs.png
"""
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

matplotlib.rcParams["font.sans-serif"] = ["Songti SC", "STHeiti", "PingFang SC", "Microsoft YaHei", "SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

FEATS = ["income", "balance", "age"]
df = pd.read_csv("data/mall_customers_sim.csv")
X = StandardScaler().fit_transform(df[FEATS])
df["cluster"] = KMeans(n_clusters=4, random_state=42, n_init=10).fit_predict(X)

pairs = [("income", "balance", "年收入（万元）", "账户平均余额（万元）"),
         ("income", "age", "年收入（万元）", "年龄（岁）"),
         ("balance", "age", "账户平均余额（万元）", "年龄（岁）")]

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
colors = ["#DD8452", "#4C72B0", "#55A868", "#C44E52"]
for ax, (fx, fy, lx, ly) in zip(axes, pairs):
    sc = ax.scatter(df[fx], df[fy], s=30, alpha=0.75, c=df["cluster"], cmap=matplotlib.colors.ListedColormap(colors))
    ax.set_title(f"{lx} × {ly}", fontsize=13)
    ax.set_xlabel(lx, fontsize=11)
    ax.set_ylabel(ly, fontsize=11)
    ax.grid(alpha=0.3)
handles = [plt.Line2D([0], [0], marker="o", linestyle="", color=colors[i],
                      markersize=8, label=f"簇 {i}（{(df['cluster']==i).sum()} 人）")
           for i in range(4)]
fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=11)
fig.suptitle("scatter 二维投影：三对特征组合 × 四簇着色（plt.scatter 的 c 参数直接用簇编号）", fontsize=15)
plt.tight_layout(rect=[0, 0.07, 1, 1])
plt.savefig("outputs/13_scatter_pairs.png", dpi=150)
print("已保存 outputs/13_scatter_pairs.png")
print("三张子图均用 plt.scatter(x, y, c=df['cluster']) 着色——簇编号即颜色编号")
