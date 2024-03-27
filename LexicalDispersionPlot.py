import jieba
from sklearn.feature_extraction.text import TfidfVectorizer

# 读取文件并获取文本内容
file_path = "C:/Users/29901/Desktop/taobao_item.txt"  # 替换为实际的文件路径
with open(file_path, 'r', encoding='utf-8') as file:
    text = file.read()

# 定义停用词列表，包含与主题关联不大的高频词
stop_words = ["非常", "不错","我","了","电动牙刷","牙刷","操作","效果","干净","可以","使用"]

# 使用jieba进行分词
seg_list = jieba.cut(text)

# 去除停用词
filtered_words = [word for word in seg_list if word not in stop_words]

# 将分词结果转化为字符串
filtered_text = " ".join(filtered_words)

# 使用TF-IDF算法提取关键词
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform([filtered_text])
feature_names = vectorizer.get_feature_names_out()

# 获取排名前五的关键词及其对应的TF-IDF值
top_keywords_indices = tfidf_matrix.toarray().argsort()[:, -8:][0]
top_keywords = [feature_names[i] for i in top_keywords_indices]
top_tfidf_scores = [tfidf_matrix.toarray()[0][i] for i in top_keywords_indices]

print("排名前八的关键词：")
for keyword, score in zip(top_keywords[::-1], top_tfidf_scores[::-1]):
    print(f"关键词: {keyword}\tTF-IDF值: {score}")




import matplotlib.pyplot as plt
from nltk import Text

# 文件路径
file_path = 'C:/Users/29901\Desktop/taobao_item.txt'

# 八个关键词列表
keywords = ['清洁', '震动', '轻便', '外观', '质量','续航','美白','品牌']

# 从文件中读取用户评论文本
with open(file_path, 'r', encoding='utf-8') as file:
    text = file.read().replace('\n', '')

# 将文本转换为nltk的Text对象
comment_text = Text(text.split())

# 根据关键词生成分布图
plt.figure(figsize=(12, 6))
comment_text.dispersion_plot(keywords)
plt.title('Frequency of Keywords in User Comments')
plt.xlabel('Word Position in the Text')
plt.ylabel('Word')
plt.show()

