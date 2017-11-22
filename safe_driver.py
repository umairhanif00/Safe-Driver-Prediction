import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
data=pd.read_csv('train.csv')
test_data=pd.read_csv('test.csv')

#print(list(data.columns.values)) # Finding cols names

data=data.drop(['ps_ind_02_cat',  'ps_ind_04_cat', 'ps_ind_06_bin', 'ps_ind_07_bin', 'ps_ind_08_bin', 'ps_ind_09_bin',
              'ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_12_bin', 'ps_ind_13_bin', 'ps_ind_14', 'ps_ind_16_bin',
             'ps_ind_18_bin', 'ps_car_02_cat', 'ps_car_04_cat', 'ps_car_05_cat', 'ps_car_06_cat',  'ps_car_08_cat',
              'ps_car_10_cat', 'ps_car_11_cat', 'ps_car_11', 'ps_car_12', 'ps_car_15', 'ps_calc_01', 'ps_calc_02',
              'ps_calc_03', 'ps_calc_04', 'ps_calc_05', 'ps_calc_06', 'ps_calc_07', 'ps_calc_08', 'ps_calc_09',
              'ps_calc_10', 'ps_calc_11', 'ps_calc_12', 'ps_calc_13', 'ps_calc_14', 'ps_calc_15_bin', 'ps_calc_16_bin',
              'ps_calc_17_bin', 'ps_calc_18_bin', 'ps_calc_19_bin', 'ps_calc_20_bin'], axis=1 )
#print(data.describe())

test_data=test_data.drop(['ps_ind_02_cat',  'ps_ind_04_cat', 'ps_ind_06_bin', 'ps_ind_07_bin', 'ps_ind_08_bin', 'ps_ind_09_bin',
              'ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_12_bin', 'ps_ind_13_bin', 'ps_ind_14', 'ps_ind_16_bin',
             'ps_ind_18_bin', 'ps_car_02_cat', 'ps_car_04_cat', 'ps_car_05_cat', 'ps_car_06_cat',  'ps_car_08_cat',
              'ps_car_10_cat', 'ps_car_11_cat', 'ps_car_11', 'ps_car_12', 'ps_car_15', 'ps_calc_01', 'ps_calc_02',
              'ps_calc_03', 'ps_calc_04', 'ps_calc_05', 'ps_calc_06', 'ps_calc_07', 'ps_calc_08', 'ps_calc_09',
              'ps_calc_10', 'ps_calc_11', 'ps_calc_12', 'ps_calc_13', 'ps_calc_14', 'ps_calc_15_bin', 'ps_calc_16_bin',
              'ps_calc_17_bin', 'ps_calc_18_bin', 'ps_calc_19_bin', 'ps_calc_20_bin'], axis=1 )


y=data['target'].values
X=data.drop('target', axis=1).values

gnb=GaussianNB()
gnb.fit(X,y)
prediction=gnb.predict(test_data)
print(gnb.score(test_data,prediction))

final_df=pd.DataFrame([test_data['id'],prediction],index=['id', 'target']).T
print(final_df)
final_df.to_csv("C:/Users/HP/Documents/DataScience/Kaggle/Safe Driver Prediction/final_df.csv") 