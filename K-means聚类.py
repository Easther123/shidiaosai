from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import FactorAnalysis

# 读取数据
file_path = r"C:/Users/29901/Desktop/原始数据_消费升级背景下电动牙刷在武汉市的市场调查_副本7.xlsx"
df = pd.read_excel(file_path)

# 数据预处理 - 筛选问卷结果，并选择需要的变量
processed_data = df[['14.  您了解的电动牙刷系统有', '11.请根据以下信息，对电动牙刷的下列作用 进行评价', '13.  请根据您的实际态度，表明您对下列说法的认同程度', '10.  请根据下列说法，判断您对电动牙刷的 满意程度','25.  您是否愿意向身边人分享电动牙刷信息']]  # 将问题1-5作为需要分析的变量

# 缺失值处理
processed_data = processed_data.dropna()

# 数据归一化
scaler = MinMaxScaler()
processed_data = scaler.fit_transform(processed_data)


# 进行因子分析，得到最终变量类群

fa = FactorAnalysis(n_components=3, random_state=0)   # 设置潜在因素的数量为3
fa.fit(processed_data)

# 获取因子负荷矩阵
loadings = fa.components_.T

# 打印因子负荷矩阵
print('因子负荷矩阵：')
print(loadings)


# 获取因子负荷矩阵
loadings = fa.loadings_

# 打印因子负荷矩阵
print('因子负荷矩阵：')
print(loadings)

# 使用熵权法对各变量类群中的子项使用熵值法客观赋权



# 自定义函数实现熵权法
def entropy_weight(data):
    # 标准化数据
    data = data / np.sum(data, axis=0)

    # 计算指标权重
    entropy = -np.sum(data * np.log(data), axis=0)
    weights = (1 - entropy) / np.sum(1 - entropy)

    return weights


# 使用熵权法对各变量类群中的子项使用熵值法客观赋权
weights = entropy_weight(processed_data)

# 打印变量权重
print('变量权重：')
print(weights)

# 轮廓系数评估聚类结果并确定最佳聚类数目
silhouette_scores = []
k_values = range(2, 11)  # 设置K的范围为2到10
for k in k_values:
    kmeans = KMeans(n_clusters=k)
    labels = kmeans.fit_predict(processed_data)
    silhouette_scores.append(silhouette_score(processed_data, labels))

best_k = silhouette_scores.index(max(silhouette_scores)) + 2  # 获取最佳聚类数目

# 使用K均值聚类算法对样本集进行聚类
kmeans = KMeans(n_clusters=best_k)
labels = kmeans.fit_predict(processed_data)

# 使用主成分分析进行降维
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(processed_data)

# 可视化聚类结果
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('K-means Clustering Results')
plt.show()

