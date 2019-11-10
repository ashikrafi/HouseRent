import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost
from scipy.stats import pearsonr
from sklearn import linear_model, model_selection
from sklearn.metrics import explained_variance_score
from sklearn.model_selection import train_test_split

data = pd.read_csv("E:/Kaggle/Competition/HouseRent/kc_house_data.csv")

features = data.iloc[:, 3:].columns.tolist()
target = data.iloc[:, 2].name

correlations = {}
for f in features:
    data_temp = data[[f, target]]
    x1 = data_temp[f].values
    x2 = data_temp[target].values
    key = f + ' vs ' + target
    correlations[key] = pearsonr(x1, x2)[0]

data_correlations = pd.DataFrame(correlations, index=['Value']).T
data_correlations.loc[data_correlations['Value'].abs().sort_values(ascending=False).index]

y = data.loc[:, ['sqft_living', 'grade', target]].sort_values(target, ascending=True).values
x = np.arange(y.shape[0])
print(y)
'exec(%matplotlib inline)'

plt.subplot(3, 1, 1)
plt.plot(x, y[:, 0])
plt.title('Sqft and Grade vs Price')
plt.ylabel('Sqft')

plt.subplot(3, 1, 2)
plt.plot(x, y[:, 1])
plt.ylabel('Grade')

plt.subplot(3, 1, 3)
plt.plot(x, y[:, 2], 'r')
plt.ylabel("Price")

regr = linear_model.LinearRegression()
new_data = data[
    ['sqft_living', 'grade', 'sqft_above', 'sqft_living15', 'bathrooms', 'view', 'sqft_basement', 'lat', 'waterfront',
     'yr_built', 'bedrooms']]

X = new_data.values
y = data.price.values

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

regr.fit(X_train, y_train)
regr.score(X_test, y_test)


# Let's try XGboost algorithm to see if we can get better results
xgb = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
                           colsample_bytree=1, max_depth=7)

traindf, testdf = train_test_split(X_train, test_size = 0.3)
xgb.fit(X_train,y_train)

predictions = xgb.predict(X_test)
print(explained_variance_score(predictions,y_test))
