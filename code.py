# Libraries & Packages:
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor


# Data Preprocessing:
sales_dataset = pd.read_csv('Train-Set.csv')
sales_dataset.head()

# Show NO of rows, col in loan_dataset
sales_dataset.shape
# Show statistical Measure
sales_dataset.describe()

# Show No of Missing Values in Each col
sales_dataset.isnull().sum()
# Dropping the missing values
sales_dataset = sales_dataset.dropna()
# No of Missing Values in Each col
sales_dataset.isnull().sum()

# Convert categorical col to numerical Val
sales_dataset.replace({'FatContent' : {'Low Fat': 0, 'LF':0, 'low fat': 0, 'Regular': 1, 'reg':1}, 'OutletSize' : {'Small': 0, 'Medium': 1, 'High': 2},
                       'LocationType': {'Tier 1': 0, 'Tier 2': 1, 'Tier 3': 2}, 
                       'OutletType': {'Grocery Store': 0, 'Supermarket Type1': 1, 'Supermarket Type2': 2, 'Supermarket Type3': 3}}, inplace=True)

sales_dataset.head()

# Separating into Data, Label
X_sales = sales_dataset.drop(columns= ['ProductID','ProductType', 'OutletID', 'OutletSales'], axis=1)
Y_sales = sales_dataset['OutletSales'] 
print(X_sales)
print(Y_sales)

# Split Data into Training, and Test
X_train, X_test, Y_train, Y_test = train_test_split(X_sales, Y_sales, test_size = 0.3, random_state = 0)
print(X_sales.shape, X_train.shape, X_test.shape)


# Support Vector Regression Algorithm:

sales_svr = svm.SVR(kernel = 'linear')
sales_svr.fit(X_train, Y_train)

y_pred = sales_svr.predict(X_test) 

# adding the list to the dataframe as column using assign(column_name = data)
dataframe_svr = pd.DataFrame(X_test)
sales_svr_df = dataframe_svr.assign(predictions = y_pred)

sales_svr_df


# Random Forest Algorithm:

sales_rf = RandomForestRegressor(random_state = 0)
sales_rf.fit(X_train, Y_train)

Y_pred = sales_rf.predict(X_test) 

# adding the list to the dataframe as column using assign(column_name = data)
dataframe_rf = pd.DataFrame(X_test)
sales_rf_df = dataframe_rf.assign(predictions = Y_pred)

sales_rf_df


# Test Data Preprocessing:
test_dataset = pd.read_csv('Test-Set.csv')
test_dataset.head()

# Show NO of rows, col in loan_dataset
test_dataset.shape

# Show statistical Measure
test_dataset.describe()

# Show No of Missing Values in Each col
test_dataset.isnull().sum()
# Dropping the missing values
test_dataset = test_dataset.dropna()
# No of Missing Values in Each col
test_dataset.isnull().sum()

# Convert categorical col to numerical Val
test_dataset.replace({'FatContent' : {'Low Fat': 0, 'LF':0, 'low fat': 0, 'Regular': 1, 'reg':1}, 'OutletSize' : {'Small': 0, 'Medium': 1, 'High': 2},
                       'LocationType': {'Tier 1': 0, 'Tier 2': 1, 'Tier 3': 2}, 
                       'OutletType': {'Grocery Store': 0, 'Supermarket Type1': 1, 'Supermarket Type2': 2, 'Supermarket Type3': 3}}, inplace=True)

test_dataset.head()
test_dataset = test_dataset.drop(columns= ['ProductID','ProductType', 'OutletID'], axis=1)
print(test_dataset)


# Prediction Using SVR
pred_test_svr = sales_svr.predict(test_dataset)

# Adding the list to the dataframe as column using assign(column_name = data)
dataframe = pd.DataFrame(test_dataset)
new_testdataset = dataframe.assign(predictions = pred_test_svr)

print("\nSVR Prediction: ") 
new_testdataset


# Prediction Using RF
pred_test_rf = sales_rf.predict(test_dataset)

# Adding the list to the dataframe as column using assign(column_name = data)
dataframe2 = pd.DataFrame(test_dataset)
newtestdataset = dataframe2.assign(predictions = pred_test_rf)

print("\nRF Prediction: ") 
newtestdataset
