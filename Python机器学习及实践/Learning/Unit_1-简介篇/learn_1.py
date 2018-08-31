#================================================================
# 良\恶性乳腺肿瘤预测
#----------------------------------------------------------------
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np 

df_train = pd.read_csv('breast-cancer-train.csv')
df_test = pd.read_csv('breast-cancer-test.csv')
# 选取'Clump Thickness'与'Cell Size'作为特征 构建测试集中的正负分类样本
df_test_negative = df_test.loc[df_test['Type']==0][['Clump Thickness','Cell Size']] # 0-良性
df_test_positive = df_test.loc[df_test['Type']==1][['Clump Thickness','Cell Size']]

# 绘制 test_set中的样本点
plt.scatter(df_test_negative['Clump Thickness'],df_test_negative['Cell Size'], marker='o', s=200, c='red')
plt.scatter(df_test_positive['Clump Thickness'],df_test_positive['Cell Size'], marker='x', s=150, c='black')
plt.xlabel('Clump Thickness')
plt.ylabel('Cell Size')

# random 随机采样直线的截距和系数
# intercept = np.random.random([1])
# coef = np.random.random([2])
# lx = np.arange(0,12)
# ly = (-intercept-lx*coef[0])/coef[1]
# plt.plot(lx,ly, c='yellow') # 画初始的随机线

# 导入sklearn中的逻辑斯蒂回归分类器
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
# 使用前10条训练样本学习直线的系数和截距
lr.fit(df_train[['Clump Thickness','Cell Size']][:300],df_train['Type'][:300])
print('Testing accuracy(10 training samples):',lr.score(df_test[['Clump Thickness','Cell Size']],df_test['Type']))

intercept = lr.intercept_
coef = lr.coef_[0, :]
lx = np.arange(0,12)
ly = (-intercept-lx*coef[0])/coef[1]
plt.plot(lx,ly, c='green')

plt.show()
#----------------------------------------------------------------


