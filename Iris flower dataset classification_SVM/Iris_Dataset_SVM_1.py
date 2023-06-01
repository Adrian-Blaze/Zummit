#IMPORT ALL NECESSARY DEPENDENCIES


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from  sklearn import linear_model, model_selection
from  sklearn.datasets import load_iris
from sklearn.svm import SVC

#LOAD THE IRIS FLOWER DATASET
iris = load_iris()
print(dir(iris))
#print(iris.feature_names)

#LOAD DATASET INTO A PANDAS DATAFRAME
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

#CONVERT TARGET NAMES TO NUMERIC REPRESENTATION
df['target_names'] = df.target.apply(lambda x : iris.target_names[x])
print(df.head())
df1 = df[df['target']==0]
df2 = df[df['target']==1]
df3 = df[df['target']==2]
print(df3.head())

#VISUALIZE SEPAL PROPERTIES FOR THE DIFFERENT CATEGORIES OF FLOWERS
plt.scatter(df1['sepal length (cm)'], df1['sepal width (cm)'], c='blue', marker='.')
plt.scatter(df2['sepal length (cm)'], df2['sepal width (cm)'], c='green', marker='+')
plt.scatter(df3['sepal length (cm)'], df3['sepal width (cm)'], c='red', marker='o')
plt.show()

#SPLIT INTO TESTING AND TRAINING DATA
x_train,x_test,y_train,y_test = model_selection.train_test_split(iris.data, iris.target, test_size=0.2)
model = SVC()
model.fit(x_train, y_train)
print(model.score(x_test, y_test))

