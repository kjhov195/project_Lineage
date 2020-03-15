#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


def best_score_scale(pred_df, true_label, scales, amt_thresholds, ceilings):
    scores = []
    param_list = []
    for i, scale in enumerate(scales):
        for j, ths in enumerate(amt_thresholds):
            for k, ceil in enumerate(ceilings): 
                pred_df2 = pred_df.copy()
                pred_df2['amount_spent'] = pred_df2['amount_spent'].map(lambda x: scale*x if x >= ths and x <= ceil else x)
                score = score_function_3(pred_df2, true_label)[0]
                scores.append(score)
                param = '(scale: '+str(scale)+' / '+'threshold: '+str(ths) +' / '+'ceiling: '+str(ceil)+')'
                param_list.append(param)
    max_score = max(scores)
    max_score_index = scores.index(max_score)
    print('The best scale, threshold and ceiling are: ', param_list[max_score_index])
    return max_score, param_list[max_score_index]


# In[3]:


def find_best_threshold2(real_test_data, pred_df, sur_pred, ams_pred, threshold_scope):
    
    mix_clf_df = pred_df.copy()
    true_label_val = return_true_label(real_test_data[['survival_time', 'amount_spent']], real_test_data)
    
    score_list = []
    sur_64_threshold_list = []
    ams_0_threshold_list = []
    for sur_64_threshold in threshold_scope:
        for ams_0_threshold in threshold_scope:
                   
            sur_pred['adj__pred_survival_yn'] = pd.DataFrame(np.where(sur_pred['survival_yn_prob_1']>sur_64_threshold, 1, 0))
            ams_pred['adj__pred_amount_yn'] = pd.DataFrame(np.where(ams_pred['amount_yn_prob_0']>ams_0_threshold, 0, 1)) 

            #mix_clf_df['survival_time'] = np.where(sur_pred['adj__pred_survival_yn']==1, 64, pred_df['survival_time'])
            #mix_clf_df['amount_spent'] = np.where(ams_pred['adj__pred_amount_yn']==0, 0, pred_df['amount_spent'])
            
            #mix_clf_df['survival_time'] = np.where(sur_pred['adj__pred_survival_yn']==1, 64, pred_df['survival_time'])
            mix_clf_df['survival_time'] = np.where(sur_pred['adj__pred_survival_yn']==1, 64, np.where(pred_df['survival_time']>64, 64, np.where(pred_df['survival_time']<1, 1, pred_df['survival_time'])))
            
            #mix_clf_df['amount_spent'] = np.where(ams_pred['adj__pred_amount_yn']==0, 0, pred_df['amount_spent'])
            mix_clf_df['amount_spent'] = np.where(ams_pred['adj__pred_amount_yn']==0, 0, np.where(pred_df['amount_spent']<0, 0, pred_df['amount_spent']))

            score = score_function_3(mix_clf_df, true_label_val)[0]
            score_list.append(score)
            
            sur_64_threshold_list.append(sur_64_threshold)
            ams_0_threshold_list.append(ams_0_threshold)
    
    max_score = max(score_list)
    max_score_index = score_list.index(max_score)
    
    best_sur_64_threshold = sur_64_threshold_list[max_score_index]
    best_ams_0_threshold = ams_0_threshold_list[max_score_index]

    sur_pred['adj__pred_survival_yn'] = pd.DataFrame(np.where(sur_pred['survival_yn_prob_1']>best_sur_64_threshold, 1, 0))
    ams_pred['adj__pred_amount_yn'] = pd.DataFrame(np.where(ams_pred['amount_yn_prob_0']>best_ams_0_threshold, 0, 1)) 

    #mix_clf_df['survival_time'] = np.where(sur_pred['adj__pred_survival_yn']==1, 64, pred_df['survival_time'])
    mix_clf_df['survival_time'] = np.where(sur_pred['adj__pred_survival_yn']==1, 64, np.where(pred_df['survival_time']>64, 64, np.where(pred_df['survival_time']<1, 1, pred_df['survival_time'])))
    
    #mix_clf_df['amount_spent'] = np.where(ams_pred['adj__pred_amount_yn']==0, 0, pred_df['amount_spent'])
    mix_clf_df['amount_spent'] = np.where(ams_pred['adj__pred_amount_yn']==0, 0, np.where(pred_df['amount_spent']<0, 0, pred_df['amount_spent']))
    

    return max_score, best_sur_64_threshold, best_ams_0_threshold, mix_clf_df


# In[4]:


def lgbm_optimization_reg(cv_splits):
    def function(num_leaves, max_depth, colsample_bytree, subsample, reg_lambda, reg_alpha):
        return cross_val_score(
               lgb.LGBMRegressor(
                   n_estimators = 1000,
                   learning_rate = 0.01,
                   num_leaves = int(num_leaves), 
                   objective = 'rmse',
                   max_depth=int(max(max_depth,1)),
                   subsample = max(subsample, 0), # Bootstrap sampling observations for each tree
                   colsample_bytree = max(colsample_bytree, 0), # Sampling features for each tree
                   reg_lambda = max(reg_lambda, 0), # L2 regularization term
                   reg_alpha = max(reg_alpha, 0), # L1 regularization term
                   n_jobs=-1, 
                   random_state=42),  
               X=X_train, 
               y=y_train, 
               cv=cv_splits,
               scoring="neg_mean_squared_error",
               n_jobs=-1).mean()

    parameters = {"num_leaves": (31, 1000),
                  "max_depth": (5, 100),
                  "subsample": (0.6, 1),
                  'colsample_bytree': (0.2, 0.9), 
                  "reg_lambda": (0, 10),
                  'reg_alpha': (0, 10)}
    
    return function, parameters


# In[ ]:


#Bayesian optimization
def bayesian_optimization(dataset, function, parameters):
    X_train, y_train, X_test, y_test = dataset
    #X_train, y_train = dataset
    n_iterations = 5
    gp_params = {"alpha": 1e-4}

    BO = BayesianOptimization(function, parameters, random_state=0)
    BO.maximize(n_iter=n_iterations, **gp_params)

    return BO.max


# In[ ]:


def xgb_evaluate(max_depth, gamma, subsample, colsample_bytree, min_child_weight):
    params = {'nthread': -1,
              'objective': "count:poisson",
              'max_depth': (int(max_depth)),
              'subsample': subsample,
              'eta': 0.1,
              'min_child_weight' : min_child_weight,
              'gamma': gamma,
              'colsample_bytree': colsample_bytree}
    # Used around 1000 boosting rounds in the full model
    cv_result = xgb.cv(params, dtrain, num_boost_round=1000, nfold=5, early_stopping_rounds=50)    
    
    # Bayesian optimization only knows how to maximize, not minimize, so return the negative RMSE
    return -cv_result.iloc[:,2].iloc[-1]


# In[ ]:


def score_function_3(predict_label, actual_label):
    
    predict = predict_label[:]
    actual = actual_label[:]

    predict.acc_id = predict.acc_id.astype('int')
    predict = predict.sort_values(by =['acc_id'], axis = 0) # 예측 답안을 acc_id 기준으로 정렬 
    predict = predict.reset_index(drop = True)
    actual.acc_id = actual.acc_id.astype('int')
    actual = actual.sort_values(by =['acc_id'], axis = 0) # 실제 답안을 acc_id 기준으로 정렬
    actual =actual.reset_index(drop=True)
    
    if predict.acc_id.equals(actual.acc_id) == False:
        print('acc_id of predicted and actual label does not match')
        sys.exit() # 예측 답안의 acc_id와 실제 답안의 acc_id가 다른 경우 에러처리 
    else:
            
        S, alpha, L, sigma = 30, 0.01, 0.1, 15  
        cost, gamma, add_rev = 0,0,0 
        profit_result = []
        survival_time_pred = list(predict.survival_time)
        amount_spent_pred = list(predict.amount_spent)
        survival_time_actual = list(actual.survival_time)
        amount_spent_actual = list(actual.amount_spent)    
        for i in range(len(survival_time_pred)):
            if survival_time_pred[i] == 64 :                 
                cost = 0
                optimal_cost = 0
            else:
                cost = alpha * S * amount_spent_pred[i]                    #비용 계산
                optimal_cost = alpha * S * amount_spent_actual[i]          #적정비용 계산 
            
            if optimal_cost == 0:
                gamma = 0
            elif cost / optimal_cost < L:
                gamma = 0
            elif cost / optimal_cost >= 1:
                gamma = 1
            else:
                gamma = (cost)/((1-L)*optimal_cost) - L/(1-L)              #반응률 계산
            
            if survival_time_pred[i] == 64 or survival_time_actual[i] == 64:
                T_k = 0
            else:
                T_k = S * np.exp(-((survival_time_pred[i] - survival_time_actual[i])**2)/(2*(sigma)**2))    #추가 생존기간 계산
                
            add_rev = T_k * amount_spent_actual[i]                         #잔존가치 계산
    
           
            profit = gamma * add_rev - cost                                #유저별 기대이익 계산
            profit_result.append(profit)
            
        score = sum(profit_result)                                         #기대이익 총합 계산
        #print(score)
    return score, profit_result


# In[ ]:


# Train data(40000, 995 + 2)
merged_data = pd.read_csv("../../preprocess/train_preprocess_1.csv").copy().sort_values('acc_id').reset_index(drop=True)
#merged_data = pd.read_csv("../../preprocess/train_preprocess_1.csv").copy().sort_values('acc_id').reset_index(drop=True).iloc[:200,:]
train_label = pd.read_csv('../../raw/train_label.csv')
merged_data = merged_data.merge(train_label , on = 'acc_id')
# Test1 data(40000, 995)
test_1 = pd.read_csv("../../preprocess/test1_preprocess_1.csv").copy().sort_values('acc_id').reset_index(drop=True)
# Test2 data(40000, 995)
test_2 = pd.read_csv("../../preprocess/test2_preprocess_1.csv").copy().sort_values('acc_id').reset_index(drop=True)


user_feature = ['acc_id'] # User id
label_feature = ['survival_time', 'amount_spent', 'survival_yn', 'amount_yn'] # Labels for profit modeling
#category = [i for i in merged_data.columns.values if ('common_item_sell'in i)|('common_item_buy'in i)|('sell_time'in i)|('buy_time'in i)|('sell_type'in i)|('buy_type'in i)|('last_sell_item_type'in i)|('last_buy_item_type'in i)]
remove_features = ['combat_days']+[i for i in merged_data.columns.values if ('day_1_' in i)|('day_4_' in i)|('day_8_' in i)|('day_17' in i)|('day_20' in i)|('day_21' in i)|('day_22' in i)|('day_23' in i)|('day_24' in i)|('day_25' in i)]
features = sorted(list(set(merged_data.columns) - set(user_feature+label_feature+remove_features)))
#scale_features = sorted(list(set(features)-set(category)))


### (1) Screening 'days' that distorting the time patterns of train&test data
merged_data['survival_yn'] = np.where(merged_data['survival_time']==64, 1, 0)
merged_data['amount_yn'] = np.where(merged_data['amount_spent']==0, 0, 1)

### (2) Robust scaling for train, test1 and test2 data
all_data = pd.concat([merged_data, test_1, test_2], sort = True).reset_index(drop=True)
transformer = RobustScaler().fit(all_data[features])
merged_data[features] = transformer.transform(merged_data[features])
test_1[features] = transformer.transform(test_1[features])
test_2[features] = transformer.transform(test_2[features])


# In[ ]:


### (3) Make Fold

merged_X = merged_data.drop(['survival_time', 'amount_spent'], axis = 1)
merged_y = merged_data[['acc_id','survival_time', 'amount_spent']] #
X1, X_fold1, y1, y_fold1 = train_test_split(merged_X, merged_y, random_state = 0, test_size = 0.2)
X2, X3, y2, y3 = train_test_split(X1, y1, random_state = 0, test_size = 0.5)
X_fold2, X_fold3, y_fold2, y_fold3 = train_test_split(X2, y2, random_state = 0, test_size = 0.5)
X_fold4, X_fold5, y_fold4, y_fold5 = train_test_split(X3, y3, random_state = 0, test_size = 0.5)

X_train1, y_train1 = pd.concat([X_fold2,X_fold3,X_fold4,X_fold5], axis = 0, sort = True).reset_index(drop=True), pd.concat([y_fold2,y_fold3,y_fold4,y_fold5], axis = 0, sort = True).reset_index(drop=True)
X_test1, y_test1 = X_fold1, y_fold1
X_train2, y_train2 = pd.concat([X_fold1,X_fold3,X_fold4,X_fold5], axis = 0, sort = True).reset_index(drop=True), pd.concat([y_fold1,y_fold3,y_fold4,y_fold5], axis = 0, sort = True).reset_index(drop=True)
X_test2, y_test2 = X_fold2, y_fold2
X_train3, y_train3 = pd.concat([X_fold1,X_fold2,X_fold4,X_fold5], axis = 0, sort = True).reset_index(drop=True), pd.concat([y_fold1,y_fold2,y_fold4,y_fold5], axis = 0, sort = True).reset_index(drop=True)
X_test3, y_test3 = X_fold3, y_fold3
X_train4, y_train4 = pd.concat([X_fold1,X_fold2,X_fold3,X_fold5], axis = 0, sort = True).reset_index(drop=True), pd.concat([y_fold1,y_fold2,y_fold3,y_fold5], axis = 0, sort = True).reset_index(drop=True)
X_test4, y_test4 = X_fold4, y_fold4
X_train5, y_train5 = pd.concat([X_fold1,X_fold2,X_fold3,X_fold4], axis = 0, sort = True).reset_index(drop=True), pd.concat([y_fold1,y_fold2,y_fold3,y_fold4], axis = 0, sort = True).reset_index(drop=True)
X_test5, y_test5 = X_fold5, y_fold5

merged_train1 = X_train1.merge(y_train1, on = 'acc_id', how = 'left') # training data of fold 1
merged_test1 = X_test1.merge(y_test1, on = 'acc_id', how = 'left') # test data of fold 1
merged_train2 = X_train2.merge(y_train2, on = 'acc_id', how = 'left') # training data of fold 2
merged_test2 = X_test2.merge(y_test2, on = 'acc_id', how = 'left') # test data of fold 2
merged_train3 = X_train3.merge(y_train3, on = 'acc_id', how = 'left') # training data of fold 3
merged_test3 = X_test3.merge(y_test3, on = 'acc_id', how = 'left') # test data of fold 3
merged_train4 = X_train4.merge(y_train4, on = 'acc_id', how = 'left') # training data of fold 4
merged_test4 = X_test4.merge(y_test4, on = 'acc_id', how = 'left') # test data of fold 4
merged_train5 = X_train5.merge(y_train5, on = 'acc_id', how = 'left') # training data of fold 5
merged_test5 = X_test5.merge(y_test5, on = 'acc_id', how = 'left') # test data of fold 5


fold_lst = [[merged_train1,merged_test1],
           [merged_train2,merged_test2],
           [merged_train3,merged_test3],
           [merged_train4,merged_test4],
           [merged_train5,merged_test5],]


# # 5 fold를 통해서 Threshold를 Gridsearch

# In[ ]:


thres_lst = []
for fold_idx in range(5):
    #fold_idx = 0

    train_fold1 , test_fold1 = fold_lst[fold_idx]

    train_fold1_acc_id = train_fold1['acc_id'].reset_index(drop = True)
    test_fold1_acc_id = test_fold1['acc_id'].reset_index(drop = True)
    merged_train = merged_data[merged_data['acc_id'].isin(train_fold1_acc_id)]
    merged_test = merged_data[merged_data['acc_id'].isin(test_fold1_acc_id)]

    train_fold1_acc_id = train_fold1['acc_id'].reset_index(drop = True)
    test_fold1_acc_id = test_fold1['acc_id'].reset_index(drop = True)
    merged_train = merged_data[merged_data['acc_id'].isin(train_fold1_acc_id)]
    merged_test = merged_data[merged_data['acc_id'].isin(test_fold1_acc_id)]


    ## amount_spent lgb parameter search
    X_train, y_train = merged_train[features], merged_train['amount_spent']
    X_test, y_test = merged_test[features], merged_test['amount_spent']
    #print(datetime.now())
    dtset = X_train, y_train, X_test, y_test

    lgbm_f,lgbm_p = lgbm_optimization_reg(5)
    bayes_opt_lgbm = bayesian_optimization(dtset, lgbm_f, lgbm_p)


    LGBM_opt_train = lgb.LGBMRegressor(n_estimators=1000, 
                                        num_leaves=int(bayes_opt_lgbm['params']['num_leaves']),                                                               
                                        max_depth=int(bayes_opt_lgbm['params']['max_depth']),
                                        subsample = bayes_opt_lgbm['params']['subsample'],
                                        colsample_bytree = bayes_opt_lgbm['params']['colsample_bytree'],
                                        reg_lambda = bayes_opt_lgbm['params']['reg_lambda'],
                                        reg_alpha = bayes_opt_lgbm['params']['reg_alpha'],
                                        learning_rate = 0.01,
                                        objective = 'rmse', 
                                        boosting_type = 'gbdt',
                                        n_jobs=-1, 
                                        random_state=42)

    LGBM_opt_train.fit(X_train,y_train)
    opt_params = LGBM_opt_train.get_params()

    # make prediction
    X_train, y_train = merged_train[features], merged_train['amount_spent']
    X_test, y_test = merged_test[features], merged_test['amount_spent']

    lgbtrain = lgb.Dataset(X_train, label=y_train)
    lgbval = lgb.Dataset(X_test, label=y_test)

    num_rounds =10000
    early_stopping_rounds=100

    LGBM_opt_model = lgb.train(opt_params, lgbtrain, num_rounds, valid_sets = lgbval, early_stopping_rounds=100)

    LGBM_opt_pred = LGBM_opt_model.predict(X_test, num_iteration = LGBM_opt_model.best_iteration)
    pred = pd.DataFrame(LGBM_opt_pred).rename(columns={0:'amount_spent'})

    pred['amount_spent'] = np.where(pred['amount_spent']<0, 0, pred['amount_spent'])

    pred_scaled = pred/pred.std()

    # save model
    # save model
    #LGBM_opt_model.save_model('ams_reg_lgb_fold%d.txt' %(fold_idx +1))


    ## survival_time xgb parameter search

    under_63_train = merged_train.loc[merged_train['survival_time']!=64]

    xt, yt = under_63_train[features], under_63_train[['survival_time']]

    dtrain = xgb.DMatrix(xt, label=yt)
    xgb_bo = BayesianOptimization(xgb_evaluate, {'max_depth': (1, 1000), 
                                                 'gamma': (0, 30),
                                                 'subsample' : (0.7, 1), 
                                                 'min_child_weight':(0, 50),
                                                 'colsample_bytree': (0.2, 1)})

    # Use the expected improvement acquisition function to handle negative numbers
    # Optimally needs quite a few more initiation points and number of iterations
    xgb_bo.maximize(init_points=3, n_iter=10, acq='ei')
    group_1_bayes_opt = xgb_bo.max


    group_1_bayes_opt['params']['max_depth'] = int(group_1_bayes_opt['params']['max_depth'])
    group_1_bayes_opt['params']['eta'] = 0.1
    group_1_bayes_opt['params']['objective'] = 'count:poisson'
    xgb_opt_params = group_1_bayes_opt['params']


    under_63_train = merged_train.loc[merged_train['survival_time']!=64]
    X_train, y_train = under_63_train[features], under_63_train[['survival_time']]
    X_test, y_test = merged_test[features], merged_test[['survival_time']]

    #create a train and validation dmatrices 
    xgtrain = xgb.DMatrix(X_train, label=y_train)
    xgval = xgb.DMatrix(X_test, label=y_test)

    watchlist = [(xgtrain, 'train'),(xgval, 'val')]

    num_rounds =10000
    #early_stopping_rounds=50

    rgrs = xgb.train(xgb_opt_params, xgtrain, num_rounds, watchlist, early_stopping_rounds=100)

    #rgrs.save_model('sur_reg_xgb_fold%d.txt' %(fold_idx +1))

    pred_sur = pd.DataFrame(rgrs.predict(xgval, ntree_limit = rgrs.best_ntree_limit)).rename(columns={0:'survival_time'})

    pred_sur['survival_time'] = np.where(pred_sur['survival_time']<1, 1, pred_sur['survival_time'])

    res = pd.concat([merged_test['acc_id'].reset_index(drop=True), pred_sur, pred_scaled], 1)

    X_train_yn, y_train_yn = merged_train[features], merged_train[['survival_yn', 'amount_yn']]
    X_test_yn, y_test_yn = merged_test[features], merged_test[['survival_yn', 'amount_yn']]

    multi_rf_clf = MultiOutputClassifier(RandomForestClassifier(n_estimators=100,
                                                                random_state=0,
                                                                verbose=3,n_jobs = -1)).fit(X_train_yn, y_train_yn)  
    true_label_yn = return_true_label(y_test_yn, merged_test)
    pred_label_yn = return_multi_pred_label(multi_rf_clf, true_label_yn, X_test_yn)

    sur_pred_res = pd.concat([pd.DataFrame(multi_rf_clf.predict_proba(X_test_yn)[0]), pred_label_yn[1][['pred_survival_time', 'survival_time']]], 1)
    sur_pred_res.columns = ['survival_yn_prob_0', 'survival_yn_prob_1', 'pred_survival_yn', 'survival_yn']
    ams_pred_res = pd.concat([pd.DataFrame(multi_rf_clf.predict_proba(X_test_yn)[1]), pred_label_yn[1][['pred_amount_spent', 'amount_spent']]], 1)
    ams_pred_res.columns = ['amount_yn_prob_0', 'amount_yn_prob_1', 'pred_amount_yn', 'amount_yn']

    #joblib.dump(multi_rf_clf, 'clf_sur0_ams1.pkl')

    scope = np.linspace(0.71, 1, 30)
    true_label_val = return_true_label(merged_test[['survival_time', 'amount_spent']], merged_test) # 실제 target의 값
    best_threshhold2 = find_best_threshold2(merged_test, res, sur_pred_res, ams_pred_res, scope)
    best_pred = best_threshhold2[3] # 앞선 modeling process를 통해 예측한 amount_spent와 survival_time의 예측값

    scales = np.round(np.arange(1.1,3.1,0.1), 2)
    thresholds = np.round(np.arange(1,np.quantile(best_pred['amount_spent'], 0.99),0.1), 2)
    ceilings = [np.quantile(best_pred['amount_spent'], i) for i in np.arange(0.99, 0.999, 0.001)]
    scale = best_score_scale(best_pred, true_label_val, scales, thresholds, ceilings)

    thres_lst.append([best_threshhold2[1],  best_threshhold2[2],float(scale[1].split(' ')[1]),float(scale[1].split(' ')[4]),float(scale[1].split(' ')[7].strip(')'))])


# # Submissions

# In[ ]:


#Bayesian optimization
def bayesian_optimization_fin(dataset, function, parameters):
    X_train, y_train = dataset
    #X_train, y_train = dataset
    n_iterations = 5
    gp_params = {"alpha": 1e-4}

    BO = BayesianOptimization(function, parameters, random_state=0)
    BO.maximize(n_iter=n_iterations, **gp_params)

    return BO.max


# In[ ]:


def lgbm_optimization_reg(cv_splits):
    def function(num_leaves, max_depth, colsample_bytree, subsample, reg_lambda, reg_alpha):
        return cross_val_score(
               lgb.LGBMRegressor(
                   n_estimators = 1000,
                   learning_rate = 0.01,
                   num_leaves = int(num_leaves), 
                   objective = 'rmse',
                   max_depth=int(max(max_depth,1)),
                   subsample = max(subsample, 0), # Bootstrap sampling observations for each tree
                   colsample_bytree = max(colsample_bytree, 0), # Sampling features for each tree
                   reg_lambda = max(reg_lambda, 0), # L2 regularization term
                   reg_alpha = max(reg_alpha, 0), # L1 regularization term
                   n_jobs=-1, 
                   random_state=42),  
               X=X_train, 
               y=y_train, 
               cv=cv_splits,
               scoring="neg_mean_squared_error",
               n_jobs=-1).mean()

    parameters = {"num_leaves": (31, 1000),
                  "max_depth": (5, 100),
                  "subsample": (0.6, 1),
                  'colsample_bytree': (0.2, 0.9), 
                  "reg_lambda": (0, 10),
                  'reg_alpha': (0, 10)}
    
    return function, parameters


# In[ ]:


under_63_train = merged_data.loc[merged_data['survival_time']!=64]

xt, yt = under_63_train[features], under_63_train[['survival_time']]

dtrain = xgb.DMatrix(xt, label=yt)
xgb_bo_fin = BayesianOptimization(xgb_evaluate, {'max_depth': (1, 1000), 
                                                 'gamma': (0, 30),
                                                 'subsample' : (0.7, 1), 
                                                 'min_child_weight':(0, 50),
                                                 'colsample_bytree': (0.2, 1)})

# Use the expected improvement acquisition function to handle negative numbers
# Optimally needs quite a few more initiation points and number of iterations
xgb_bo_fin.maximize(init_points=3, n_iter=10, acq='ei')
group_1_bayes_opt_fin = xgb_bo_fin.max


group_1_bayes_opt_fin['params']['max_depth'] = int(group_1_bayes_opt_fin['params']['max_depth'])
group_1_bayes_opt_fin['params']['eta'] = 0.1
group_1_bayes_opt_fin['params']['objective'] = 'count:poisson'
xgb_opt_fin_params = group_1_bayes_opt_fin['params']

under_63_train = merged_data.loc[merged_data['survival_time']!=64]

X_train, y_train = under_63_train[features], under_63_train[['survival_time']]

#create a train and validation dmatrices 
xgtrain = xgb.DMatrix(X_train, label=y_train)

#params = {'colsample_bytree': 0.2,
#          'gamma': 10.0,
#          'max_depth': 173,
#          'min_child_weight': 0.0,
#          'subsample': 1.0,
#          'eta': 0.1,
#          'objective': 'count:poisson'}

num_rounds = 1000

rgrs_fin = xgb.train(xgb_opt_fin_params, xgtrain, num_rounds)#, early_stopping_rounds=50)
rgrs_fin.save_model('sur_reg_xgb.txt' )


sur_pred_test_1 = pd.DataFrame(rgrs_fin.predict(xgb.DMatrix(test_1[features]))).rename(columns={0:'survival_time'})
sur_pred_test_1['survival_time'] = np.where(sur_pred_test_1['survival_time']<1, 1, sur_pred_test_1['survival_time'])

sur_pred_test_2 = pd.DataFrame(rgrs_fin.predict(xgb.DMatrix(test_2[features]))).rename(columns={0:'survival_time'})
sur_pred_test_2['survival_time'] = np.where(sur_pred_test_2['survival_time']<1, 1, sur_pred_test_2['survival_time'])


# In[ ]:


X_train, y_train = merged_data[features], merged_data['amount_spent']
#X_test, y_test = merged_test[features], merged_test['amount_spent']
#print(datetime.now())
dtset = X_train, y_train

lgbm_f,lgbm_p = lgbm_optimization_reg(5)
bayes_opt_lgbm_fin = bayesian_optimization_fin(dtset, lgbm_f, lgbm_p)

print(bayes_opt_lgbm_fin)

LGBM_opt_train_fin = lgb.LGBMRegressor(n_estimators=1000, 
                                    num_leaves=int(bayes_opt_lgbm_fin['params']['num_leaves']),                                                               
                                    max_depth=int(bayes_opt_lgbm_fin['params']['max_depth']),
                                    subsample = bayes_opt_lgbm_fin['params']['subsample'],
                                    colsample_bytree = bayes_opt_lgbm_fin['params']['colsample_bytree'],
                                    reg_lambda = bayes_opt_lgbm_fin['params']['reg_lambda'],
                                    reg_alpha = bayes_opt_lgbm_fin['params']['reg_alpha'],
                                    learning_rate = 0.01,
                                    objective = 'rmse', 
                                    boosting_type = 'gbdt',
                                    n_jobs=-1, 
                                    random_state=42)

LGBM_opt_train_fin.fit(X_train, y_train)

lgbm_opt_fin_params = LGBM_opt_train_fin.get_params()

X_train, y_train = merged_data[features], merged_data['amount_spent']
#X_test, y_test = merged_test[features], merged_test['amount_spent']

lgbtrain = lgb.Dataset(X_train, label=y_train)
#lgbval = lgb.Dataset(X_test, label=y_test)

num_rounds =1000
#early_stopping_rounds=100

LGBM_opt_model_fin = lgb.train(lgbm_opt_fin_params, lgbtrain, num_rounds)

LGBM_opt_model_fin.save_model('ams_reg_lgb.txt' )

ams_pred_test_1 = pd.DataFrame(LGBM_opt_model_fin.predict(test_1[features])).rename(columns={0:'amount_spent'})
ams_pred_test_2 = pd.DataFrame(LGBM_opt_model_fin.predict(test_2[features])).rename(columns={0:'amount_spent'})

ams_pred_test_1['amount_spent'] = np.where(ams_pred_test_1['amount_spent']<0, 0, ams_pred_test_1['amount_spent'])
ams_pred_test_1_scaled = ams_pred_test_1/ams_pred_test_1.std()

ams_pred_test_2['amount_spent'] = np.where(ams_pred_test_2['amount_spent']<0, 0, ams_pred_test_2['amount_spent'])
ams_pred_test_2_scaled = ams_pred_test_2/ams_pred_test_2.std()

X_train_yn, y_train_yn = merged_data[features], merged_data[['survival_yn', 'amount_yn']]

multi_rf_clf = MultiOutputClassifier(RandomForestClassifier(n_estimators=100,
                                                            random_state=0, 
                                                            verbose=3, 
                                                            n_jobs = -1)).fit(X_train_yn, y_train_yn)  


# In[ ]:


try:
    sur_yn_ths  = pd.DataFrame(thres_lst).mean(axis=0)[0]
    ams_yn_ths  = pd.DataFrame(thres_lst).mean(axis=0)[1]
    ams_scale   = pd.DataFrame(thres_lst).mean(axis=0)[2]
    ams_ths     = pd.DataFrame(thres_lst).mean(axis=0)[3]
    ams_ceiling = pd.DataFrame(thres_lst).mean(axis=0)[4]
    print('OptPass')
except:
    sur_yn_ths = (0.95 + 0.88 + 0.9 + 0.75 + 0.82)/5
    ams_yn_ths = (0.98 + 0.98 + 0.9 + 0.94 + 0.97)/5
    ams_scale = (3.0 + 3.0 + 2.6 + 3.0 + 3.0)/5
    ams_ths = (1.6 + 4.3 + 5.0 + 2.0 + 1.4)/5
    ams_ceiling = (15.528607821434065 + 8.724613230855145 + 9.063103154425901 + 5.193386725328818 + 5.787545518172495)/5
finally:
    print(sur_yn_ths,ams_yn_ths,ams_scale,ams_ths,ams_ceiling)
    
joblib.dump([sur_yn_ths,ams_yn_ths,ams_scale,ams_ths,ams_ceiling], 'md_thres.pkl')


# In[ ]:


sur_pred_clf_test_1 = pd.DataFrame(multi_rf_clf.predict_proba(test_1[features])[0])
sur_pred_clf_test_1.columns = ['survival_yn_prob_0', 'survival_yn_prob_1']
ams_pred_clf_test_1 = pd.DataFrame(multi_rf_clf.predict_proba(test_1[features])[1])
ams_pred_clf_test_1.columns = ['amount_yn_prob_0', 'amount_yn_prob_1']

joblib.dump(multi_rf_clf, 'clf_sur0_ams1.pkl')

res_test_1 = pd.concat([test_1['acc_id'], sur_pred_test_1, ams_pred_test_1_scaled], 1)

sur_pred_clf_test_1['adj__pred_survival_yn'] = pd.DataFrame(np.where(sur_pred_clf_test_1['survival_yn_prob_1']>sur_yn_ths, 1, 0))
ams_pred_clf_test_1['adj__pred_amount_yn'] = pd.DataFrame(np.where(ams_pred_clf_test_1['amount_yn_prob_0']>ams_yn_ths, 0, 1)) 

res_test_1['survival_time'] = np.where(sur_pred_clf_test_1['adj__pred_survival_yn']==1, 64, np.where(res_test_1['survival_time']>64, 64, np.where(res_test_1['survival_time']<1, 1, res_test_1['survival_time'])))
res_test_1['amount_spent'] = np.where(ams_pred_clf_test_1['adj__pred_amount_yn']==0, 0, np.where(res_test_1['amount_spent']<0, 0, res_test_1['amount_spent']))

#res_test_1.to_csv("test1_predict.csv", index=False) # Final submission for test 1


# In[ ]:


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


# In[ ]:




