import matplotlib.pyplot as plt
import wordcloud
from wordcloud import WordCloud

# 文件路径
file_path = 'C:/Users/29901/Desktop/taobao_item--.txt'

# 读取用户评论内容
with open(file_path, 'r', encoding='utf-8') as file:
    user_comments = file.read().replace('\n', '')



# 生成词云图
wc = WordCloud(font_path ="C:/Windows/Fonts/msyh.ttc",background_color='white').generate(user_comments)

# 绘制词云图
plt.figure(figsize=(8, 8))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()
