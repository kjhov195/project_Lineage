# 더미화 할 때 사용
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
plt.rc('font', family='Malgun Gothic')

def merge_dummy_data(data, cat_features):
    for cat_feature in cat_features:
        tmp = pd.get_dummies(data[cat_feature], prefix=cat_feature, drop_first=True)
        
        data = pd.concat([data, tmp], 1)
    data = data.drop(cat_features, 1)   
    
    return data

# 더미화 하지않을 때 사용
def categorical_to_int(data, cat_features):
    for cat_feature in cat_features:
        le = LabelEncoder()
        data[cat_feature] = le.fit_transform(data[cat_feature].map(str))
        #data[cat_feature] = data[cat_feature].map({0:0, 'etc':1, 'adena':2, 'enchant_scroll':3, 'armor':4, 'accessory':5, 'weapon':6, 'spell':7})

    return data

def return_final_data(data, cat_features, dummy):
    data = data.fillna(0)
    
    if dummy == True:
        return merge_dummy_data(data, cat_features)
    else:
        return categorical_to_int(data, cat_features) 
    
def return_true_label(y_test_data, raw_test_data):
    y_true = y_test_data.reset_index(drop=True)
    true_label = pd.concat([raw_test_data['acc_id'].reset_index(drop=True), y_true], 1)
    return true_label

# score 구하기
import pandas as pd
import numpy as np
import sys

def score_function_2(predict_label, actual_label):
    
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
    return score

# 라벨 반환
def return_multi_pred_label(model, true_label, test_data):
    pred_res = pd.DataFrame(model.predict(test_data)).rename(columns={0:'survival_time', 1:'amount_spent'})
    pred_label = pd.concat([true_label['acc_id'], pred_res], 1)
    
    pred_true_df = pd.concat([pred_label, true_label.iloc[:, 1:]], 1)
    pred_true_df.columns = ['acc_id', 'pred_survival_time', 'pred_amount_spent', 'survival_time', 'amount_spent']
    return pred_label, pred_true_df

# 하이퍼파라미터 튜닝
def hypertuning_rscv(est, p_distr, nbr_iter, X_train, y_train, X_test, y_test_true_label):
    rdmsearch = RandomizedSearchCV(est, param_distributions=p_distr, n_iter=nbr_iter, verbose=10,
                                  n_jobs=-1, cv=3, scoring='neg_mean_squared_error')
    #CV = Cross-Validation ( here using Stratified KFold CV)
    rdmsearch.fit(X_train, y_train)
    ht_params = rdmsearch.best_params_
    ht_score = rdmsearch.best_score_

    pred_label = return_multi_pred_label(rdmsearch.best_estimator_, y_test_true_label, X_test)
    score = score_function_2(pred_label[0], y_test_true_label)   
    
    return ht_params, ht_score, pred_label, score, rdmsearch.best_estimator_

# multioutput feature importance plot 그리기 
def plotImp_multioutput(model, X , ord, num = 25):
    feature_imp = pd.DataFrame(sorted(zip(model.estimators_[ord].feature_importances_,X.columns)), 
                               columns=['Value','Feature'])
    plt.figure(figsize=(20, 15))
    #plt.rcParams.update({'font.size': 10})
    #plt.set(font_scale = 5)
    sns.set(font_scale=2)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title('LightGBM Features (avg over folds)')
    
    #plt.tight_layout()
    plt.show()


def find_best_threshold(real_test_data, pred_df, sur_pred, ams_pred, threshold_scope):
    
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

            score = score_function_2(mix_clf_df, true_label_val)
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
    
    figure, (ax1,ax2) = plt.subplots(nrows=1, ncols=2)
    figure.set_size_inches(10,3)
    sns.distplot(mix_clf_df['survival_time'], ax=ax1) # 예측값
    sns.distplot(mix_clf_df['amount_spent'], ax=ax2) # 예측값

    sns.distplot(real_test_data['survival_time'], ax=ax1) # true값
    sns.distplot(real_test_data['amount_spent'], ax=ax2) # true값   
    plt.show()
    
    return max_score, best_sur_64_threshold, best_ams_0_threshold, mix_clf_df