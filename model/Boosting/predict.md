

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# For adjusting 'scales' of train, test1 and test2
from sklearn.preprocessing import RobustScaler
# For hyperparameter optimization
from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score
# For amount_spent modeling
import lightgbm as lgb
# For survival_time modeling
import xgboost as xgb
# For screening survival, non-spent users modeling
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import train_test_split

# Customizing functions
from MJ_functions import merge_dummy_data, categorical_to_int, return_final_data, return_true_label, return_multi_pred_label
from MJ_functions import hypertuning_rscv, plotImp_multioutput, score_function_2, find_best_threshold

import pickle

from sklearn.externals import joblib
from joblib import dump, load
```

# Data input
전처리된 데이터를 입력받고 필요한 feature들을 선택해주는 과정, train data의 경우 label과 합쳐주기 위한 코드를 추가


```python
# Train data(40000, 995 + 2)
merged_data = pd.read_csv("../preprocess/train_preprocess_1.csv").copy().sort_values('acc_id').reset_index(drop=True)
#merged_data = pd.read_csv("../../preprocess/train_preprocess_1.csv").copy().sort_values('acc_id').reset_index(drop=True).iloc[:200,:]
train_label = pd.read_csv('../raw/train_label.csv')
merged_data = merged_data.merge(train_label , on = 'acc_id')
# Test1 data(40000, 995)
test_1 = pd.read_csv("../preprocess/test1_preprocess_1.csv").copy().sort_values('acc_id').reset_index(drop=True)
# Test2 data(40000, 995)
test_2 = pd.read_csv("../preprocess/test2_preprocess_1.csv").copy().sort_values('acc_id').reset_index(drop=True)


user_feature = ['acc_id'] # User id
label_feature = ['survival_time', 'amount_spent', 'survival_yn', 'amount_yn'] # Labels for profit modeling
#category = [i for i in merged_data.columns.values if ('common_item_sell'in i)|('common_item_buy'in i)|('sell_time'in i)|('buy_time'in i)|('sell_type'in i)|('buy_type'in i)|('last_sell_item_type'in i)|('last_buy_item_type'in i)]
remove_features = ['combat_days']+[i for i in merged_data.columns.values if ('day_1_' in i)|('day_4_' in i)|('day_8_' in i)|('day_17' in i)|('day_20' in i)|('day_21' in i)|('day_22' in i)|('day_23' in i)|('day_24' in i)|('day_25' in i)]
features = sorted(list(set(merged_data.columns) - set(user_feature+label_feature+remove_features)))
#scale_features = sorted(list(set(features)-set(category)))
```

### Binary label 추가
Binary Search를 위하여 survival_yn, amount_yn 컬럼을 추가


```python
### (1) Screening 'days' that distorting the time patterns of train&test data
merged_data['survival_yn'] = np.where(merged_data['survival_time']==64, 1, 0)
merged_data['amount_yn'] = np.where(merged_data['amount_spent']==0, 0, 1)
```

### Robust scaling 
각 데이터셋간의 scale 차이를 고려하여 Robust scaling  수행


```python
### (2) Robust scaling for train, test1 and test2 data
all_data = pd.concat([merged_data, test_1, test_2], sort = True).reset_index(drop=True)
transformer = RobustScaler().fit(all_data[features])
merged_data[features] = transformer.transform(merged_data[features])
test_1[features] = transformer.transform(test_1[features])
test_2[features] = transformer.transform(test_2[features])
```

### Model load
저장된 모델 파일을 load


```python
LGBM_opt_model_fin = lgb.Booster(model_file='../model/Boosting/ams_reg_lgb.txt')
rgrs_fin = xgb.Booster(model_file='../model/Boosting/sur_reg_xgb.txt')
multi_rf_clf = load('../model/Boosting/clf_sur0_ams1.pkl')
sur_yn_ths,ams_yn_ths,ams_scale,ams_ths,ams_ceiling = load('../model/Boosting/md_thres.pkl')
```

# Prediction
저장된 모델을 활용하여 test1 test2의 amount spent와 survival time 예측

### Survival time Regression


```python
under_63_train = merged_data.loc[merged_data['survival_time']!=64]

xt, yt = under_63_train[features], under_63_train[['survival_time']]

dtrain = xgb.DMatrix(xt, label=yt)

under_63_train = merged_data.loc[merged_data['survival_time']!=64]

X_train, y_train = under_63_train[features], under_63_train[['survival_time']]

#create a train and validation dmatrices 
xgtrain = xgb.DMatrix(X_train, label=y_train)


sur_pred_test_1 = pd.DataFrame(rgrs_fin.predict(xgb.DMatrix(test_1[features]))).rename(columns={0:'survival_time'})
sur_pred_test_1['survival_time'] = np.where(sur_pred_test_1['survival_time']<1, 1, sur_pred_test_1['survival_time'])

sur_pred_test_2 = pd.DataFrame(rgrs_fin.predict(xgb.DMatrix(test_2[features]))).rename(columns={0:'survival_time'})
sur_pred_test_2['survival_time'] = np.where(sur_pred_test_2['survival_time']<1, 1, sur_pred_test_2['survival_time'])
```

### amount spent regression


```python
X_train, y_train = merged_data[features], merged_data['amount_spent']
#X_test, y_test = merged_test[features], merged_test['amount_spent']

lgbtrain = lgb.Dataset(X_train, label=y_train)
#lgbval = lgb.Dataset(X_test, label=y_test)

num_rounds =1000
#early_stopping_rounds=100


ams_pred_test_1 = pd.DataFrame(LGBM_opt_model_fin.predict(test_1[features])).rename(columns={0:'amount_spent'})
ams_pred_test_2 = pd.DataFrame(LGBM_opt_model_fin.predict(test_2[features])).rename(columns={0:'amount_spent'})

ams_pred_test_1['amount_spent'] = np.where(ams_pred_test_1['amount_spent']<0, 0, ams_pred_test_1['amount_spent'])
ams_pred_test_1_scaled = ams_pred_test_1/ams_pred_test_1.std()

ams_pred_test_2['amount_spent'] = np.where(ams_pred_test_2['amount_spent']<0, 0, ams_pred_test_2['amount_spent'])
ams_pred_test_2_scaled = ams_pred_test_2/ams_pred_test_2.std()
```

### survival time, amount spent Classifier


```python
X_train_yn, y_train_yn = merged_data[features], merged_data[['survival_yn', 'amount_yn']]

sur_pred_clf_test_1 = pd.DataFrame(multi_rf_clf.predict_proba(test_1[features])[0])
sur_pred_clf_test_1.columns = ['survival_yn_prob_0', 'survival_yn_prob_1']
ams_pred_clf_test_1 = pd.DataFrame(multi_rf_clf.predict_proba(test_1[features])[1])
ams_pred_clf_test_1.columns = ['amount_yn_prob_0', 'amount_yn_prob_1']

res_test_1 = pd.concat([test_1['acc_id'], sur_pred_test_1, ams_pred_test_1_scaled], 1)

sur_pred_clf_test_1['adj__pred_survival_yn'] = pd.DataFrame(np.where(sur_pred_clf_test_1['survival_yn_prob_1']>sur_yn_ths, 1, 0))
ams_pred_clf_test_1['adj__pred_amount_yn'] = pd.DataFrame(np.where(ams_pred_clf_test_1['amount_yn_prob_0']>ams_yn_ths, 0, 1)) 

res_test_1['survival_time'] = np.where(sur_pred_clf_test_1['adj__pred_survival_yn']==1, 64, np.where(res_test_1['survival_time']>64, 64, np.where(res_test_1['survival_time']<1, 1, res_test_1['survival_time'])))
res_test_1['amount_spent'] = np.where(ams_pred_clf_test_1['adj__pred_amount_yn']==0, 0, np.where(res_test_1['amount_spent']<0, 0, res_test_1['amount_spent']))

#res_test_1.to_csv("test1_predict.csv", index=False) # Final submission for test 1
sur_pred_clf_test_2 = pd.DataFrame(multi_rf_clf.predict_proba(test_2[features])[0])
sur_pred_clf_test_2.columns = ['survival_yn_prob_0', 'survival_yn_prob_1']
ams_pred_clf_test_2 = pd.DataFrame(multi_rf_clf.predict_proba(test_2[features])[1])
ams_pred_clf_test_2.columns = ['amount_yn_prob_0', 'amount_yn_prob_1']

res_test_2 = pd.concat([test_2['acc_id'], sur_pred_test_2, ams_pred_test_2_scaled], 1)

sur_pred_clf_test_2['adj__pred_survival_yn'] = pd.DataFrame(np.where(sur_pred_clf_test_2['survival_yn_prob_1']>sur_yn_ths, 1, 0))
ams_pred_clf_test_2['adj__pred_amount_yn'] = pd.DataFrame(np.where(ams_pred_clf_test_2['amount_yn_prob_0']>ams_yn_ths, 0, 1)) 

res_test_2['survival_time'] = np.where(sur_pred_clf_test_2['adj__pred_survival_yn']==1, 64, np.where(res_test_2['survival_time']>64, 64, np.where(res_test_2['survival_time']<1, 1, res_test_2['survival_time'])))
res_test_2['amount_spent'] = np.where(ams_pred_clf_test_2['adj__pred_amount_yn']==0, 0, np.where(res_test_2['amount_spent']<0, 0, res_test_2['amount_spent']))

#res_test_2.to_csv("test2_predict.csv", index=False) # Final submission for test 2
```

### Scaling and return output


```python
# (scale: 3.0 / threshold: 6)
res_test_1['amount_spent'] = res_test_1['amount_spent'].map(lambda x : ams_scale*x if x >= ams_ths and x <= ams_ceiling else x)
res_test_2['amount_spent'] = res_test_2['amount_spent'].map(lambda x : ams_scale*x if x >= ams_ths and x <= ams_ceiling else x)
res_test_1.to_csv("test1_predict.csv", index=False) # Final submission for test 1
res_test_2.to_csv("test2_predict.csv", index=False) # Final submission for test 2
```
