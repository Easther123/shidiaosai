import pandas as pd
from scipy.stats import chi2_contingency

# 读取Excel文件
file_path = r"C:\Users\29901\Desktop\原始数据_消费升级背景下电动牙刷在武汉市的市场调查_副本7.xlsx"
df = pd.read_excel(file_path)

# 进行列联分析
crosstab = pd.crosstab(df['您目前从事的职业'], df['我对电动牙刷比较感兴趣 - 请根据以下信息，选择最符合您实际感受的选项'])

# 进行卡方检验
chi2, p, dof, expected = chi2_contingency(crosstab)

# 打印结果
print("列联分析结果：")
print(crosstab)
print("卡方值: ", chi2)
print("P值: ", p)
print("自由度: ", dof)


#进行列联分析
crosstab = pd.crosstab(df['请问您的月收入范围为   元'], df['我对电动牙刷比较感兴趣 - 请根据以下信息，选择最符合您实际感受的选项'])

#进行卡方检验
from scipy.stats import chi2_contingency

chi2, p, dof, expected = chi2_contingency(crosstab)

#打印结果
print("列联分析结果：")
print(crosstab)
print("卡方值: ", chi2)
print("P值: ", p)
print("自由度: ", dof)

