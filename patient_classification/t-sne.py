import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

df= pd.read_csv('/Users/zhangxiwen/Desktop/case.csv')

X = df.drop('诊断',axis=1)
y = df['诊断']
y = y.map({'CML-CP': 'CML-CP', 'Normal': 'Normal', 'AML': 'AML', 'APL': 'APL', 'ALL': 'ALL', 'CLL': 'CLL', 'MM': 'MM', 'MPN': 'MPN'})
#y = y.map({'CML-CP': 'CML-CP', 'Normal': 'Normal', 'AML': 'AML', 'APL': 'APL', 'ALL': 'ALL', 'CLL': 'CLL', 'MM': 'MM', 'MPN': 'MPN'})

X_std = StandardScaler().fit_transform(X)

tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X_std)
X_tsne_data = np.vstack((X_tsne.T, y)).T
df_tsne = pd.DataFrame(X_tsne_data, columns=['Dim1', 'Dim2', '诊断'])

plt.figure(figsize=(8, 8))
sns.scatterplot(data=df_tsne, hue='诊断', x='Dim1', y='Dim2')
plt.savefig('/Users/zhangxiwen/Desktop/result.jpg')
plt.show()
