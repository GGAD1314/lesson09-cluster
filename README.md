# lesson09-cluster · 第 9 次课作业：聚类与客户分群

《人工智能程序设计与实践》2026 秋 · 第 9 次课（11/13）
案例复现 + 自主实践 + 独立研究，全部脚本在本仓库根目录运行。

## 目录结构

```
lesson09-cluster/
├── README.md                 本文件
├── progress.md               进度报告（按课程模板）
├── AI使用记录.md              指令原文 + 验证过程
├── data/                     数据文件（不得改动，hash 校验）
│   ├── mall_customers_sim.csv     300行7列，余额万元计（起点数据）
│   └── mall_customers_yuan.csv    同批客户，余额元计（量纲陷阱数据）
├── generated/                代码与报告
│   ├── 01_load_and_viz.py         案例复现 Step1：初检 + 初始散点
│   ├── 02_kmeans_start.py         案例复现 Step2：起点版 k-均值（125/125/50）
│   ├── 02b_yuan_rerun.py          案例复现 Step3：元计翻车（141/111/48）
│   ├── 02c_scaled_kmeans.py       案例复现 Step4：标准化修正（126/124/50）
│   ├── 03_choose_k.py             案例复现 Step5：肘部法+轮廓系数选 k
│   ├── 04_iteration_demo.py       案例复现 Step6：质心迭代 2×2 过程图
│   ├── 11_age_recluster.py        自主实践①：加 age 重画像（k=4，四中文命名）
│   ├── 12_seed_stability.py       自主实践②：三种子稳定性（42/7/2026）
│   ├── 13_scatter_pairs.py        自主实践③：三对特征 scatter 二维着色
│   ├── 21_mapreduce_kmeans.py     独立研究①：MapReduce 并行 k-均值模拟
│   ├── 22_dbscan_ring.py          独立研究②：环形数据 KMeans 失败 vs DBSCAN
│   ├── 23_silhouette.py           独立研究③：轮廓系数手算 + sklearn
│   ├── research_notes.md          独立研究三题文字结论
│   ├── mall_report.md             （运行时生成）案例分群报告
│   └── age_report.md              （运行时生成）加 age 后四簇画像报告
└── outputs/                  全部图表（150 dpi，中文标题）
```

## 运行方法

```bash
pip install pandas scikit-learn matplotlib   # 环境：Python 3.9+
python generated/01_load_and_viz.py          # 按编号顺序逐个运行
```

## 检查点读数（教师验收对照）

| 检查点 | 读数 | 出处 |
|---|---|---|
| 起点版各簇人数 | **125 / 125 / 50** | 02_kmeans_start.py |
| 元计翻车各簇人数 | **141 / 111 / 48** | 02b_yuan_rerun.py |
| 标准化修正后 | **126 / 124 / 50** | 02c_scaled_kmeans.py |
| 肘部法拐点 | **k=3**（inertia 600→147.9→62.8→45.2…） | 03_choose_k.py |
| 轮廓系数最高 | **k=2（0.7351）**，k=3 为 0.6902——两证据不一致，业务取 k=3 | 03_choose_k.py |
| 质心迭代 | **3 轮收敛**，inertia 79.16→62.79 单调下降 | 04_iteration_demo.py |
| 四簇人数（加 age，k=4） | **121 / 57 / 72 / 50** | 11_age_recluster.py |
| 四中文命名 | 低收低余的年轻客群 / 中收中余的新锐白领客群 / 中收中余的中年主力客群 / 高收高余的富裕客群 | 11_age_recluster.py |
| 三种子稳定性 | 42/7/2026 两两标签一致率 **100%**，画像不变 | 12_seed_stability.py |
| MapReduce vs sklearn | 标签一致率 **100%**，inertia 均 104.93 | 21_mapreduce_kmeans.py |
| 环形数据 | KMeans 切两半失败；DBSCAN 完美分双环（400/400） | 22_dbscan_ring.py |
| 轮廓系数手算 | 单点 s=0.4813；全局 0.5693 | 23_silhouette.py |

## 数据完整性（hash 校验，红线：数据文件不得改动）

```
SHA256
13DC15593DB7EBD5349982AE83EC2439DEC9461ACDD3984493F1E10C84EEDF9F  data/mall_customers_sim.csv
060D349F4E7FD44DF6B97BF76C5876FF38A7570075C491730CF9361B4344EA07  data/mall_customers_yuan.csv
```

## 红线遵守声明

1. customer_id 绝不进模型——全部聚类只用 income / balance / age 特征列；
2. 距离类模型先标准化（StandardScaler）——02c 起所有聚类均标准化后进行；
3. random_state=42、n_init=10 显式写——保证可复现；数据文件未改动（见上方 hash）。
