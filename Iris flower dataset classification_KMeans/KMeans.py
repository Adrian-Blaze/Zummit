import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from  sklearn import linear_model, model_selection
from  sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
iris = load_iris()
print(dir(iris))
#print(iris.target_names)
df = pd.DataFrame(iris.data, columns=iris.feature_names)

df.drop(['sepal length (cm)', 'sepal width (cm)'], axis='columns', inplace=True)

#plt.scatter(df[['petal length (cm)']],df[['petal width (cm)']], marker='.', c='red' )
#plt.show()
model = KMeans(n_clusters=3)
cluster = model.fit_predict(df)
df['cluster'] = cluster
#print(df.head(20))
df1 = df[df.cluster == 0]
df2 = df[df.cluster == 1]
df3 = df[df.cluster == 2]
print(df3.head())
#plt.scatter(df1[['petal length (cm)']],df1[['petal width (cm)']], marker='.', c='red')
#plt.scatter(df2[['petal length (cm)']],df2[['petal width (cm)']], marker='.', c='blue')
#plt.scatter(df3[['petal length (cm)']],df3[['petal width (cm)']], marker='.', c='black')
#plt.show()
scaler = MinMaxScaler()
df['Petal length scaled'] = scaler.fit_transform(df[['petal length (cm)']])
df['Petal width scaled'] = scaler.fit_transform(df[['petal width (cm)']])
print(df.head(10))
cluster = model.fit_predict(df[['Petal length scaled', 'Petal width scaled']])
df['cluster2'] = cluster
#print(df.head(20))
df1 = df[df.cluster2 == 0]
df2 = df[df.cluster2 == 1]
df3 = df[df.cluster2 == 2]
#print(df3.head())
plt.scatter(df1[['Petal length scaled']],df1[['Petal width scaled']], marker='.', c='red')
plt.scatter(df2[['Petal length scaled']],df2[['Petal width scaled']], marker='.', c='blue')
plt.scatter(df3[['Petal length scaled']],df3[['Petal width scaled']], marker='.', c='black')
plt.show()
