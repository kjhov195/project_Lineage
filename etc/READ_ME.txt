[OS]
Windows 10 EDU

[Aanaconda]
4.7.10.

[Python]
3.7.3

[Modules]
xgboost 0.90
lightgbm 2.2.3
bayesian-optimization 1.0.1
lime 0.1.1.36

[코드 실행 순서 및 방법]

1. raw 폴더에 제공 데이텅 총 16개 csv파일을 넣어준다
(required - /raw folder)
train_activity.csv
train_combat.csv    
train_payment.csv  
train_pledge.csv    
train_trade.csv      
train_label.csv   

test1_activity.csv
test1_combat.csv  
test1_payment.csv 
test1_pledge.csv  
test1_trade.csv    

test2_activity.csv
test2_combat.csv  
test2_payment.csv 
test2_pledge.csv  
test2_trade.csv    

(output)
.
2. preprocess 폴더에 있는 preprocess.py를 실행해준다
- raw데이터들을 불러와 전처리를 수행해주는 코드
(required - /raw folder)
/raw/train_activity.csv
/raw/train_combat.csv    
/raw/train_payment.csv  
/raw/train_pledge.csv    
/raw/train_trade.csv      
/raw/train_label.csv   

/raw/test1_activity.csv
/raw/test1_combat.csv  
/raw/test1_payment.csv 
/raw/test1_pledge.csv  
/raw/test1_trade.csv    

/raw/test2_activity.csv
/raw/test2_combat.csv  
/raw/test2_payment.csv 
/raw/test2_pledge.csv  
/raw/test2_trade.csv    

(output - /preprocess )
train_preprocess_1.csv
test1_preprocess_1.csv
test2_preprocess_1.csv

3-1. /model/boosting 폴더에 있는 create_mdoel.py를 실행해준다
- Bayesian optimizer를 통해서 survival_time을 예측하기 위한 xgboost regression의 hyperparameter를 찾고 모델을 train 해준 sur_reg_xgb.txt 파일
- Bayesian optimizer를 통해서 amount_spent를 예측하기 위한 lightgbm regression의 hyperparameter를 찾고 모델을 train 해준 ams_reg_lgb.txt 파일
- survival_time이 64인지 classification을 한 random forest 모델과 amount_spent가 0인지 classification을 한 random forest 모델을 train해준 clf_sur0_ams1.pkl 파일
- 최종 결과를 제출하기위해 survival_time을 64로 보내주는 기준, amount_spent를 0으로 보내주는 기준, amount_spent의 상위값을 보정해주는 기준을 grid search를 통해 찾아서 저장한 md_thres.pkl 파일
- 총 4개의 모델저장파일을 만들어주는 코드

(required)
/model/Boosting/MJ_functions.py
/preprocess/train_preprocess_1.csv
/preprocess/test1_preprocess_1.csv
/preprocess/test2_preprocess_1.csv

(output - /model/boosting)
ams_reg_lgb.txt
sur_reg_xgb.txt
clf_sur0_ams1.pkl
md_thres.pkl

4. /predict 폴더에 있는 predict.py를 실행해준다
- create_mdoel.py에서 생성된 4가지 모델 파일을 불러와서 test데이터르 활용한 prediction 값 test1_predict.csv, test2_predict.csv 을 만들어주는 코드

(required)
/predict/MJ_functions.py
/model/Boosting/ams_reg_lgb.txt
/model/Boosting/sur_reg_xgb.txt
/model/Boosting/clf_sur0_ams1.pkl
/model/Boosting/md_thres.pkl

(output - /predict)
test1_predict.csv
test2_predict.csv

