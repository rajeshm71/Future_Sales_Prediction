import os
import pandas as pd
import numpy as np

#Read Dataset
df = pd.read_csv('dataset.csv')
df.head()

#print("Total no. of datapoints",df.shape[0])
#print("Total no. of datapoints",df.shape[1])

import datetime

#Create Features using 'DATE' Column
df['Date'] = pd.to_datetime(df['DATE'])
#df['Date'] = df['Date'].dt.strftime('%d.%m.%Y')
df['year'] = pd.DatetimeIndex(df['Date']).year
df['month'] = pd.DatetimeIndex(df['Date']).month
df['day'] = pd.DatetimeIndex(df['Date']).day
df['dayofyear'] = pd.DatetimeIndex(df['Date']).dayofyear
df['weekofyear'] = pd.DatetimeIndex(df['Date']).weekofyear
df['weekday'] = pd.DatetimeIndex(df['Date']).weekday
df['quarter'] = pd.DatetimeIndex(df['Date']).quarter
df['is_month_start'] = pd.DatetimeIndex(df['Date']).is_month_start
df['is_month_end'] = pd.DatetimeIndex(df['Date']).is_month_end
df = df.drop(['Date','DATE'], axis = 1)
df.head()

# import warnings
# warnings.filterwarnings("ignore")
# import matplotlib
# matplotlib.use(u'nbAgg')
# import matplotlib.pyplot as plt
# import seaborn as sns
# %matplotlib inline

#one hot encoding of categorical features
df = pd.get_dummies(df, columns=['is_month_start'], drop_first=False, prefix='m_start')
df = pd.get_dummies(df, columns=['is_month_end'], drop_first=False, prefix='m_end')

#split train-test data (most recent data is split into train and test)
train=df.iloc[:138,:]
test=df.iloc[138:,:]


#seperate target variable from feature
y_train=train['SALES'].values
#y_val=val['SALES'].values
y_test=test['SALES'].values

X_train = train.drop(['SALES'], axis = 1)
#val = val.drop(['SALES'], axis = 1)
X_test = test.drop(['SALES'], axis =1)



#Machine learning Models
#Random Forest
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from datetime import datetime
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint
from scipy.stats import uniform

#Cross Validation
start=datetime.now()
param_dist = {"n_estimators":sp_randint(75,500),
              "max_depth": sp_randint(10,15),
              "min_samples_split": sp_randint(2,14),
              "min_samples_leaf": sp_randint(1,20)}

clf = RandomForestRegressor(random_state=25,n_jobs=-1)

rf_random = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=5,cv=10,scoring='neg_root_mean_squared_error',random_state=20)

rf_random.fit(X_train,y_train)
#print('mean test scores',rf_random.cv_results_['mean_test_score'])
#print('mean train scores',rf_random.cv_results_['mean_train_score'])
#print('Total time taken is {}'.format(datetime.now()-start))

#fit the data using best parameters obtained using cross validation
rfreg=RandomForestRegressor(n_estimators=332,max_depth=12,min_samples_leaf=12, min_samples_split=12,oob_score=True, n_jobs=-1,random_state=20)
rfreg.fit(X_train,y_train)
y_train_pred = rfreg.predict(X_train)
y_test_pred = rfreg.predict(X_test)
pred_train_rf= rfreg.predict(X_train)
print("Train Error",np.sqrt(mean_squared_error(y_train,pred_train_rf)))
pred_test_rf = rfreg.predict(X_test)
print("Test Error",np.sqrt(mean_squared_error(y_test,pred_test_rf)))



#XGBoost
from xgboost import XGBRegressor
n_estimators = sp_randint(75,500) #[100,200,300,400,500,600,700,800,900]
max_depth = sp_randint(2,15) #[2,4,6,8,10,12,14,16,18]
params = {"n_estimators":n_estimators,"max_depth":max_depth}
xgb = XGBRegressor()
rsm = RandomizedSearchCV(xgb,params,cv=5,scoring='neg_root_mean_squared_error',n_jobs=-1)
rsm.fit(X_train,y_train)
print("Best parameter obtained from RandomSearch CV: \n", rsm.best_params_)
print("Best Score : ", rsm.best_score_)

xgb=XGBRegressor(n_estimators=450,max_depth=4, n_jobs=-1)
xgb.fit(X_train,y_train)
y_train_pred = xgb.predict(X_train)
y_test_pred = xgb.predict(X_test)

pred_train_rf= xgb.predict(X_train)
print("Train Error for XGboost",np.sqrt(mean_squared_error(y_train,pred_train_rf)))
pred_test_rf = xgb.predict(X_test)
print("Test Error for XGboost",np.sqrt(mean_squared_error(y_test,pred_test_rf)))

#make predictions
import argparse
parser=argparse.ArgumentParser(description='Enter number of days')
parser.add_argument('-n',required=True,help='number of future dates',default=1)
args=parser.parse_args()

def predict(n):
    try:
        def create_features(datelist):
            df=pd.DataFrame()
            df['Date']=datelist
            df['Date'] = pd.to_datetime(df['Date'])
            #df['Date'] = df['Date'].dt.strftime('%d.%m.%Y')
            df['year'] = pd.DatetimeIndex(df['Date']).year
            df['month'] = pd.DatetimeIndex(df['Date']).month
            df['day'] = pd.DatetimeIndex(df['Date']).day
            df['dayofyear'] = pd.DatetimeIndex(df['Date']).dayofyear
            df['weekofyear'] = pd.DatetimeIndex(df['Date']).weekofyear
            df['weekday'] = pd.DatetimeIndex(df['Date']).weekday
            df['quarter'] = pd.DatetimeIndex(df['Date']).quarter
            df['is_month_start'] = pd.DatetimeIndex(df['Date']).is_month_start
            df['is_month_end'] = pd.DatetimeIndex(df['Date']).is_month_end
            return df

        def create_dummies(df):
            df = pd.get_dummies(df, columns=['is_month_start'], drop_first=False, prefix='m_start')
            df = pd.get_dummies(df, columns=['is_month_end'], drop_first=False, prefix='m_end')
            return df

        datelist=pd.date_range(start="2019-03-26",periods=n).to_pydatetime().tolist()
        datelist=[date.strftime("%Y-%m-%d") for date in datelist]
        df=create_features(datelist)
        df=create_dummies(df)
        date=df['Date'].values
        df = df.drop(['Date'], axis = 1)
        train_features=list(X_train.columns.values)
        test_features=list(df.columns.values)
        for tr_feat in train_features:
            if tr_feat not in test_features:
                idx=train_features.index(tr_feat)
                df.insert(loc=idx,column=tr_feat,value=[0]*len(df))
        predictions = rfreg.predict(df)
        predictions=predictions.astype('int32')
        df=pd.DataFrame()
        df['Date']=date
        df['Predictions']=predictions
        return df
    except:
        print("Please enter number of days greater than zero")
predictions=predict(int(args.n))
if predictions is not None:
    print(predictions)
