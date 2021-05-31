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
features = pd.DataFrame(boston.data,columns=boston.feature_names)
target = pd.DataFrame(boston.target,columns=['MEDV'])
df = pd.concat([features,target],axis=1)

# calculate correlation between every column on the data
corr = df.corr(method='pearson')

#take absolute values of correlation
corrs = [abs(corr[attr]['MEDV']) for attr in list(features)]

#make a list of pairs [(features),corrs]
l = list(zip(list(features),corrs))

with open("correlation.txt", 'w') as outfile:
        outfile.write("list of pairs [(features),corrs]: %s\n" % l)

#sort the list of pairs in reverse order woth correlation values as the key of sorting
l.sort(key=lambda x:x[0], reverse= True)

#unzip the pairs to two lists 
# zip(*l) takes a list that looks like [[a,b,c],[d,e,f],[g,h,i]]and returns [[a,d,g],[b,e,h],[c,f,i]]
labels,corrs = list(zip(*l))

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

# plot the distribution of the target variable "MEDV"
plt.xlabel('target')
sns.set(rc={'figure.figsize':(11.7,8.27)})
save = sns.distplot(target, bins=30)
plt.savefig("target",dpi=120)
plt.show()
plt.close()

# Train and Test the data Model
X = features
Y = target
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=5)

#Fit a model on the train section 
#Set random seed
seed = 42
regr = RandomForestRegressor(max_depth=2, random_state=seed)
regr.fit(X_train, Y_train)

# Report training set score
train_score = regr.score(X_train, Y_train) * 100
# Report test set score
test_score = regr.score(X_test, Y_test) * 100

with open("metrics.txt", 'w') as outfile:
        outfile.write("Training variance explained: {%s}\n" % train_score)
        outfile.write("Test variance explained: {%s}\n" % test_score)

# Linear Regrassion
lin_model = LinearRegression()
lin_model.fit(X_train, Y_train)

# model evaluation for training set
y_train_predict = lin_model.predict(X_train)
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
