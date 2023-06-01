#IMPORT ALL NECESSARY DEPENDENCIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from  sklearn import linear_model, model_selection
from  sklearn.datasets import load_iris

#LOAD THE IRIS FLOWER DATASET
iris = load_iris()
print(dir(iris))

x_train,x_test,y_train,y_test = model_selection.train_test_split(iris.data, iris.target, test_size=0.2)


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=1000)
model.fit(x_train, y_train)
print(model.score(x_test, y_test))
