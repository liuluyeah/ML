import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest,f_classif
import matplotlib.pyplot as plt
from sklearn import cross_validation

train=pd.read_csv('D:/aselfstudy/ML/train.csv')
test=pd.read_csv('D:/aselfstudy/ML/test.csv')
Y = train['Target']
#特征选择算法
predictors=['Feature_1','Feature_2','Feature_2','Feature_3','Feature_4','Feature_5','Feature_6','Feature_7']
#选择交叉验证，fit模型
selector=SelectKBest(f_classif,k=5)
selector.fit(train[predictors],Y)
scores=-np.log10(selector.pvalues_)
#加上一下画图
plt.bar(range(len(predictors)),scores)
plt.xticks(range(len(predictors)),predictors,rotation='vertical')
plt.show()

print("Training the LR model ...")
model = LogisticRegression(random_state=1)
model.fit(X_train, Y)
#归一化处理
#打印每一列的最大值、最小值、均值
#print(X_train.max(axis=0))
#print(X_train.min(axis=0))
#print(X_train.mean(axis=0))
mean_val = X_train.mean(axis=0)
max_val = X_train.max(axis=0)
min_val = X_train.min(axis=0)
row = X_train.shape[0]
column = X_train.shape[1]
for rn in range(row):
    for cn in range(column):
        X_train[rn][cn] = (X_train[rn][cn] - mean_val[cn])/(max_val[cn] - min_val[cn])
print('归一化完成')

kf=cross_validation.KFold(X_train.shape[0],n_folds=5,random_state=1)
scores=cross_validation.cross_val_score(model,X_train,Y,cv=kf)
print(scores.mean())

## Predict the Competition Data with the newly trained model
print("Predicting the Competition Data...")
y_test = model.predict_proba(X_test) # Predict the Target, getting the probability.
pred = y_test[:, 1]                  # Get the probabilty of being 1.
pred_df = pd.DataFrame(data={'Target':pred})
submissions = pd.DataFrame(ID).join(pred_df)   
##Write the CSV File and Get Ready for Submission
# Save the predictions out to a CSV file
#print("Writing predictions ······")
submissions.to_csv("D:/aselfstudy/ML/result.csv", index=False)
print("finished")
