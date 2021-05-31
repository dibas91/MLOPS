import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

boston = load_boston()
#print(boston.DESCR)
features = pd.DataFrame(boston.data,columns=boston.feature_names)
#features['CRIM']
target = pd.DataFrame(boston.target,columns=['MEDV'])
#print(max(target['target']))
#print(min(target['target']))
df = pd.concat([features,target],axis=1)
df.describe() # to describe
# calculate correlation between every column on the data
corr = df.corr(method='pearson')
#corr

#take absolute values of correlation

corrs = [abs(corr[attr]['MEDV']) for attr in list(features)]

#make a list of pairs [(corr,features)]

l = list(zip(corrs,list(features)))
#sort the list of pairs in reverse order woth correlation values as the key of sorting
l.sort(key=lambda x:x[0], reverse= True)
#l
#unzip the pairs to two lists
#zip(*l) takes a list that looks like [[a,b,c],[d,e,f],[g,h,i]] and returns [[a,d,g],[b,e,h],[c,f,i]]

corrs,labels = list(zip(*l))

#corrs

labels

#plot correlation with respect to the target variable as a bar graph

index = np.arange(len(labels))
plt.figure(figsize =(15,5))

plt.bar(index,corrs,width=0.5)

plt.xlabel('features')
plt.ylabel("correlation with target values")
plt.xticks(index,labels)
plt.savefig("feature",dpi=120) 
plt.show()
plt.close()

plt.xlabel('target')
sns.set(rc={'figure.figsize':(11.7,8.27)})
save = sns.distplot(target, bins=30)
plt.savefig("target",dpi=120)
plt.show()
plt.close()

X = pd.DataFrame(boston.data,columns=boston.feature_names)
#X = pd.DataFrame(np.c_[boston['CRIM'], boston['RM']], columns = ['CRIM','RM'])
Y = target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=5)

# print(X_train.shape)
# print(X_test.shape)
# print(Y_train.shape)
# print(Y_test.shape)

#Fit a model on the train section
#Set random seed
seed = 42
regr = RandomForestRegressor(max_depth=2, random_state=seed)
regr.fit(X_train, Y_train)

# Report training set score
train_score = regr.score(X_train, Y_train) * 100
# Report test set score
test_score = regr.score(X_test, Y_test) * 100

# print("Training variance explained: {%s}\n" % train_score)

# print("Test variance explained: {%s}\n" % test_score)


with open("metrics.txt", 'w') as outfile:
        outfile.write("Training variance explained: {%s}\n" % train_score)
        outfile.write("Test variance explained: {%s}\n" % test_score)

lin_model = LinearRegression()
lin_model.fit(X_train, Y_train)

# model evaluation for training set
y_train_predict = lin_model.predict(X_train)
#print(y_train_predict[:5])
rmse_training = (np.sqrt(mean_squared_error(Y_train, y_train_predict)))
r2_training = r2_score(Y_train, y_train_predict)


# model evaluation for testing set
y_test_predict = lin_model.predict(X_test)
rmse_testing = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))
r2_testing = r2_score(Y_test, y_test_predict)

with open("evalution.txt", 'w') as outfile:
        outfile.write("model evaluation for rmse_training set: {%s}\n" % rmse_training)
        outfile.write("model evaluation for r2_training set: {%s}\n" % r2_training)
        outfile.write("model evaluation for rmse_testing set: {%s}\n" % rmse_testing)
        outfile.write("model evaluation for r2_testing set: {%s}\n" % r2_testing)
