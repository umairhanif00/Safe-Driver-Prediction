import pandas as pd
import tensorflow as tf
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense

data=pd.read_csv('train.csv')
test_data=pd.read_csv('test.csv')
#n_cols=data.shape[1]
data=data.drop(['ps_ind_02_cat',  'ps_ind_04_cat', 'ps_ind_06_bin', 'ps_ind_07_bin', 'ps_ind_08_bin', 'ps_ind_09_bin',
              'ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_12_bin', 'ps_ind_13_bin', 'ps_ind_14', 'ps_ind_16_bin',
             'ps_ind_18_bin', 'ps_car_02_cat', 'ps_car_04_cat', 'ps_car_05_cat', 'ps_car_06_cat',  'ps_car_08_cat',
              'ps_car_10_cat', 'ps_car_11_cat', 'ps_car_11', 'ps_car_12', 'ps_car_15', 'ps_calc_01', 'ps_calc_02',
              'ps_calc_03', 'ps_calc_04', 'ps_calc_05', 'ps_calc_06', 'ps_calc_07', 'ps_calc_08', 'ps_calc_09',
              'ps_calc_10', 'ps_calc_11', 'ps_calc_12', 'ps_calc_13', 'ps_calc_14', 'ps_calc_15_bin', 'ps_calc_16_bin',
              'ps_calc_17_bin', 'ps_calc_18_bin', 'ps_calc_19_bin', 'ps_calc_20_bin'], axis=1 )

test_data=test_data.drop(['ps_ind_02_cat',  'ps_ind_04_cat', 'ps_ind_06_bin', 'ps_ind_07_bin', 'ps_ind_08_bin', 'ps_ind_09_bin',
              'ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_12_bin', 'ps_ind_13_bin', 'ps_ind_14', 'ps_ind_16_bin',
             'ps_ind_18_bin', 'ps_car_02_cat', 'ps_car_04_cat', 'ps_car_05_cat', 'ps_car_06_cat',  'ps_car_08_cat',
              'ps_car_10_cat', 'ps_car_11_cat', 'ps_car_11', 'ps_car_12', 'ps_car_15', 'ps_calc_01', 'ps_calc_02',
              'ps_calc_03', 'ps_calc_04', 'ps_calc_05', 'ps_calc_06', 'ps_calc_07', 'ps_calc_08', 'ps_calc_09',
              'ps_calc_10', 'ps_calc_11', 'ps_calc_12', 'ps_calc_13', 'ps_calc_14', 'ps_calc_15_bin', 'ps_calc_16_bin',
              'ps_calc_17_bin', 'ps_calc_18_bin', 'ps_calc_19_bin', 'ps_calc_20_bin'], axis=1 )

predictor_features=data.drop(['target'], axis=1).as_matrix()
target_var=to_categorical(data.target)
#print(test_data.shape)
test_data=test_data.as_matrix()

model = Sequential()
model.add(Dense(100, activation='relu', input_shape=(15,)))
model.add(Dense(100, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(predictor_features,target_var)
prediction=model.predict(test_data)
#prediction=prediction.astype(int)
probability_true = prediction[:,1]
print(probability_true)
final_df=pd.DataFrame([test_data['id'],prediction], index=['id', 'target']).T
print(final_df)
final_df.to_csv("C:/Users/HP/Documents/DataScience/Kaggle/Safe Driver Prediction/NN_final_df.csv")
for p in prediction:
    print("Prediction is: ",p)