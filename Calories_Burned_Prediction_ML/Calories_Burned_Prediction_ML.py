import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import metrics
from sklearn.svm import SVC
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
  
import warnings
warnings.filterwarnings('ignore')

# Merging Datasets
df_cal = pd.read_csv('calories.csv')
df_exer = pd.read_csv('exercise.csv')
df = pd.merge(df_exer,df_cal)

# EDA Analysis
sb.scatterplot(x=(df['Weight']), y=(df['Height']))
#plt.show()

features = ['Age', 'Height', 'Weight', 'Duration']

plt.subplots(figsize=(15, 20))
for i, col in enumerate(features):
    plt.subplot(2, 2, i + 1)
    x = df.sample(1000)
    sb.scatterplot(x=x[col],y=x['Calories'])
plt.tight_layout()
#plt.show()

features = df.select_dtypes(include='float').columns

plt.subplots(figsize = (15, 20))
for i, col in enumerate(features):
    plt.subplot(2, 3, i + 1)
    sb.displot(df[col])
plt.tight_layout()
#plt.show()

df.replace({
    'male' : 0,
    'female' : 1
}, inplace=True)

plt.figure(figsize=(8,8))
sb.heatmap(df.corr() > 0.9, annot=True, cbar=False)
#plt.show()

remove_lst = ['Weight', 'Duration']
df.drop(remove_lst, axis=1, inplace=True)


# Data normilization and Model Training
features = df.drop(['User_ID', 'Calories'], axis=1)
target = df['Calories'].values

X_train, X_val, Y_train, Y_val = train_test_split(features, target, test_size=0.15, random_state=22)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.fit_transform(X_val)

from sklearn.metrics import mean_absolute_error as mae
XGBR = XGBRegressor()
XGBR.fit(X_train, Y_train)
train_preds = XGBR.predict(X_train)
val_preds = XGBR.predict(X_val)

print('Training Error: ', mae(Y_train, train_preds))
print('Validation Error: ', mae(Y_val, val_preds))