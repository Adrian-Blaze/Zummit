# IMPORT ALL THE NECESSARY PACKAGES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from  sklearn import linear_model, model_selection
from sklearn.preprocessing import MinMaxScaler
import re

# READ THE DATA TO WORK WITH INTO A DATAFRAME

df = pd.read_csv('bengaluru_house_prices.csv')
print(df.head(10))

#INSPECTING THE DATA FOR DATA WRANGLING

#print(df.shape)
#print(df.describe())
df2 = df.drop(['area_type', 'society','bath', 'balcony'],axis=1)
#print(df2.head(10))
#print(df2.groupby('availability')['availability'].agg('count')) #CHECK UNIQUE VALUE COUNT
#print(df2.isnull().sum()) #CHECK NA VALUE COUNT
df3 = df2.dropna()
df4 = df3.copy()
df4.size = df3['size'].apply(lambda x : int(x.split(' ')[0]))
#print(df4.head(10))
#print(df4['total_sqft'].unique())

#CREATION OF FUNCTIONS TO AID IN DATA CLEANING

def is_float(x): #FUNCTION TO IDENTIFY FLOATING POINTS NUMBER WHEN APPLIED ON A DATAFRAME
    try:
        x = float(x)

    except:
        return False
    return True
    
#a = df4[~df4['total_sqft'].apply(is_float)]
#print(a.head(10))

def total_sqft_range(x): # FUNCTION THAT ENSURES UNIFORM UNIT FOR THE SIZE COLUMN
    Regex = re.compile(r"(\d+(?:\.\d+)?)\s*(\w+)")
    mo2 = Regex.search(x)
    Regex2 = re.compile(r'(\d+)\s-\s(\d+)')
    c = Regex2.search(x)
    if c != None:
         x = (float(c.group(1)) + float(c.group(2)))/2
    elif mo2 != None:
        if mo2.group(2) == 'Perch':
            x = float(272.25) * float(mo2.group(1))
        elif mo2.group(2) == 'Sq':
            x = float(10.764) * float(mo2.group(1))
        elif mo2.group(2) == 'Acres':
            x = float(43560) * float(mo2.group(1))
        elif mo2.group(2) == 'Cents':
            x = float(435.56) * float(mo2.group(1))
        elif mo2.group(2) == 'Guntha':
            x = float(1089) * float(mo2.group(1))
        elif mo2.group(2) == 'Grounds':
            x = float(2400.35203) * float(mo2.group(1))
    
    try:
        x = float(x)
        return x
    except:
        return None
    
    
df4['total_sqft2'] = df4['total_sqft'].apply(total_sqft_range)
#print(df4)

#d = df4[~df4['total_sqft2'].apply(is_float)]
#print(d)
#print(df4.loc[9423])
df5 = df4.drop(['total_sqft'], axis=1)
#print(df5.head(10))
#print(df5['location'].describe())
df5.location = df5.location.apply(lambda x: x.strip())
#print(df5['location'].describe())
location_stats = df5.groupby('location')['location'].agg('count').sort_values(ascending=False)
#print(location_stats)
#print(df5.isnull().sum())
#nan = pd.isnull(df5['total_sqft2'])
#print(df5[nan])
#print(location_stats[location_stats<=10 ])
other_location = location_stats[location_stats<=10]
df5['location'] = df5.location.apply(lambda x: 'other' if x in other_location else x)
df6 = df5.copy()
#print(df6.head(10))
#print(len(df6['availability'].unique()))
df6['availability'] = df6.availability.apply(lambda x: 'unavailable' if x != 'Ready To Move' else x)
#print(df6.head(10))
#print(df6['availability'].unique())

#APPLY GET DUMMIES TO CONVERT TO CATEGORICAL VARIIABLE TYPE WHICH IS NEEDED

dummies1 = pd.get_dummies(df6.availability)
dummies2 = pd.get_dummies(df6.location)
#print(dummies1)
df7 = pd.concat([df6, dummies1.drop('unavailable', axis=1)], axis=1)
df7 = pd.concat([df7, dummies2.drop('other',axis=1)], axis=1)
#print(df7.head(10))
df7['totalsqft_per_size'] = df7.total_sqft2/df7['size']
#print(df7.head(10))
#print(df7['totalsqft_per_size'].describe())
mean = df7['totalsqft_per_size'].mean()
std =  df7['totalsqft_per_size'].std()
upper = mean + 1*std
lower = mean - 1*std
df8 = df7[~((df7['totalsqft_per_size'] < lower) | (df7['totalsqft_per_size'] > upper))]
#print(df8.shape)
#print(df8[df8['totalsqft_per_size'] < 300])
df9 = df8[~(df8['totalsqft_per_size'] < 300)]
#print(df9.shape)

# OUTLIER DETECTION AND REMOVAL WITH DIFFERENT PLOTS

def outlier_removal(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        #print(key)
        #print(subdf)
        avg = subdf['totalsqft_per_size'].mean()
        std = subdf['totalsqft_per_size'].std()
        upperlimit = avg + 2*std
        lowerlimit = avg - 2*std
        #print (upperlimit, lowerlimit)
        #print(type(upperlimit))
        df_out1 = subdf[~((subdf['totalsqft_per_size'] < lowerlimit) | (subdf['totalsqft_per_size'] > upperlimit))]
        df_out = pd.concat([df_out, df_out1], ignore_index=True)
    return df_out
    
df10 = outlier_removal(df9)

scaler = MinMaxScaler() # APPLICATION OF SCALING FOR BETTER MODEL PERFORMANCE
df10['price_per_sqft'] = df10['price']* 100000 / df10['total_sqft2']
#print(df10.shape)

def plot(idf, location):
   
    bd2 = idf[(idf.location == location) & (idf['size']==2)]
    bd3 = idf[(idf.location == location) & (idf['size']==3)]
    #print(idf)
    #print(bd3)
    if bd2.shape[0] == 0:
        print("No data for 2 bed properties")
    else:
        plt.scatter(bd2[['total_sqft2']], bd2[['price']], c='blue', s=50, label = '2 bed')
    if bd3.shape[0] == 0:
        print("No data for 3 bed properties")
    else:
        plt.scatter(bd3[['total_sqft2']], bd3[['price']], c='green', s=50, marker='+', label = '3 bed')
    plt.xlabel('total square feet')
    plt.ylabel('price_per_sqft')
    plt.title(location)
    plt.legend()
    return plt.show()
#plot(df10, 'Hebbal')

def cost_per_size_outlier_removal(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        size_stats = {}
        for size, size_df in location_df.groupby('size'):
            size_stats[size] = {'mean': np.mean(size_df.price_per_sqft), 'std': np.std(size_df.price_per_sqft), 'count': size_df.shape[0]}
        #print(size_stats)
        for size, size_df in location_df.groupby('size'):
            stats = size_stats.get(size-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, size_df[size_df.price_per_sqft<(stats['mean'])].index.values)
    #print(size_stats)
    return df.drop(exclude_indices, axis='index')

df11 = cost_per_size_outlier_removal(df10)
#print(df11.shape)
df12 = df11.drop(['totalsqft_per_size', 'price_per_sqft'], axis=1)
print(df12.head())
#plot(df12, 'Hebbal')

#INITIALIZING TARGET OR DEPENDENT VARIABLE AND THE INPUT FEATURES

X = df12.drop(['price', 'location', 'availability'], axis=1)
y = df12.price

#Split the data into the training and testing set

x_train,x_test,y_train,y_test = model_selection.train_test_split(X, y, test_size=0.2)

# Machine Learning model using Linear Regression

lin_model = linear_model.LinearRegression(normalize=True)
lin_model.fit(x_train, y_train)
print(lin_model.score(x_test, y_test))


