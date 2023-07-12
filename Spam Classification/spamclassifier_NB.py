#IMPORT NECESSARY DEPENDENCIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from  sklearn import linear_model, model_selection
from  sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from  sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

#READ IN DATA FROM CSV FILE
df = pd.read_csv(f'C:/Users/user/Desktop/spam.csv')
#print(df.describe())
#dummies = pd.get_dummies(df.Category)
#print(dummies)

#CONVERT TEXT DATA TO NUMERIC DATA
le =LabelEncoder()
df.Category = le.fit_transform(df.Category)
print(df)
#v = CountVectorizer()
#X = v.fit_transform(df.Message)

#SPLIT DATASET INTO TRAINING AND TEST DATA
x_train,x_test,y_train,y_test = model_selection.train_test_split(df.Message, df.Category, test_size=0.2)
#model = MultinomialNB()
#model.fit(x_train, y_train)
#print(model.score(x_test, y_test))

#INITIALIZING PREDICTION TESTING RANDOM INPUT
Email = 'Hello, bobby. You have won big. Send your details for redemption of gift'
#Email = v.transform(Email)
#print(model.predict(Email))

#ALLIGN STEPS IN PIPELINE AND TRAIN MODEL ON DATA
model2 = Pipeline([('a', CountVectorizer()), ('b', MultinomialNB())])
model2.fit(x_train, y_train)
print(model2.predict_proba([Email]))
print(model2.predict([Email]))
