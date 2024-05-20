import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from xgboost import plot_importance
import matplotlib.pyplot as plt
from tqdm import tqdm  # 进度条库

# 读取训练数据
train = pd.read_csv("input/tr_FE.csv")

# 分离特征和标签
y_train = train['click']
X_train = train.drop(['click'], axis=1)

# 检查和转换所有 object 类型的列为 category 类型
for col in X_train.select_dtypes(include=['object']).columns:
    X_train[col] = X_train[col].astype('category')


def modelfit(alg, dtrain, predictors, target, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()  # 参数
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)  # 训练集数据与标签

        # 使用 tqdm 进度条
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          metrics='auc', early_stopping_rounds=early_stopping_rounds,
                          verbose_eval=True, show_stdv=False, as_pandas=True, seed=27,
                          callbacks=[tqdm()])

        alg.set_params(n_estimators=cvresult.shape[0])

    alg.fit(dtrain[predictors], dtrain[target], eval_metric='auc')

    # Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:, 1]

    # Print model report:
    print("Model Report")
    print("Accuracy : %.4g" % accuracy_score(dtrain[target].values, dtrain_predictions))
    print("AUC Score (Train): %f" % roc_auc_score(dtrain[target], dtrain_predprob))

    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    plt.show()


# 使用 XGBClassifier 模型
model = XGBClassifier(n_estimators=350, max_depth=10, objective='binary:logistic', min_child_weight=50,
                      subsample=0.8, gamma=0, learning_rate=0.2, colsample_bytree=0.5, seed=27,
                      enable_categorical=True)

# 训练模型
model.fit(X_train, y_train)

# 绘制特征重要性
plot_importance(model, importance_type="gain")
plt.show()

# 获取特征重要性
features = X_train.columns
feature_importance_values = model.feature_importances_

feature_importances = pd.DataFrame({'feature': list(features), 'importance': feature_importance_values})
feature_importances.sort_values('importance', inplace=True, ascending=False)
print(feature_importances)

print(model.get_booster().get_score(importance_type="gain"))

# 保存特征重要性到文件
feature_importances.to_csv('feature.csv', index=False)
