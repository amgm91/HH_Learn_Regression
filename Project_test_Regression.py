import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import cross_val_score, cross_val_predict

# -----------------------------------------------------------------------------
# Load the data
Data = loadmat('ChemTrainNew.mat')

#print(Data.keys())

# .mat file keys
# 'XtrainDS' --> Training Input data
# 'YtrainDS' --> Training Output data
# 'XtestDS' --> Testing Input data

chem_in = pd.DataFrame(Data['XtrainDS'])
chem_out = pd.DataFrame(Data['YtrainDS'])
chem_test_in = pd.DataFrame(Data['XtestDS'])

print()

# chem_in -> 4466 samples * 65 variable --> Training Input data
# chem_out -> 4466 samples * 1 variable --> Training Output data

# -----------------------------------------------------------------------------------

# Standrdize the input using z-score

chem_in_std = chem_in.apply(zscore)

# -----------------------------------------------------------------------------------

# drawing scatter plots for each variable and output

#for i in range(len(chem_in.columns)):
#    plt.scatter(chem_in_std[i], chem_out, label="variable %d and output" %i)
#    plt.xlabel(("var %d" % i))
#    plt.ylabel("output")
#    plt.show()


#chem_in_std.values.to_list
#print(type(chem_in_std[0].tolist()))

#chem_in_out.append(chem_in_std.iloc[:, :])

#chem_in_out.append(chem_out.iloc[:, 0])
#print(pd.DataFrame(chem_in_out))

# calculate covariance between each variable and output variable

chem_in_out =[]
for i in range(len(chem_in_std.columns)):
    chem_in_out.append(chem_in_std.iloc[:, i])

chem_in_out.append(chem_out.iloc[:, 0])
chem_in_out = pd.DataFrame(chem_in_out)
chem_in_out = chem_in_out.reset_index(drop=True)
chem_in_out = chem_in_out.transpose()
chem_in_out_cov = chem_in_out.cov()
# List of covariance between every variable and output variable
chem_cov = chem_in_out_cov.iloc[0:-1, -1]
chem_cov = chem_cov.abs()
chem_cov = chem_cov.tolist()
chem_cov = list(enumerate(chem_cov))
chem_cov.sort(key=lambda tup: tup[1], reverse=True)
#print(chem_cov)

# Sorting variables by covariance value
chem_in_sort = []
for i in range(len(chem_in_std.columns)):
    index = chem_cov[i][0]
    chem_in_sort.append(chem_in_std.iloc[:, index])
chem_in_sort = pd.DataFrame(chem_in_sort)
chem_in_sort = chem_in_sort.reset_index(drop=True)
chem_in_sort = chem_in_sort.transpose()
#print(chem_in_sort.shape)

# -----------------------------------------------------------------------------------

# Constructing Linear Regression model


#chem_x_train, chem_x_test, chem_y_train, chem_y_test = \
#        train_test_split(chem_in_sort, chem_out, train_size=0.7, test_size=0.3)

#reg_mdl = reg_mdl.fit(chem_x_train, chem_y_train)
#pred_train = reg_mdl.predict(chem_x_train)
#pred_test = reg_mdl.predict(chem_x_test)
# calculate RMSE of train and test prediction
#RMSE_train = np.sqrt(metrics.mean_squared_error(chem_y_train, pred_train))
#RMSE_test = np.sqrt(metrics.mean_squared_error(chem_y_test, pred_test))
#print("Out mean: ", chem_out.mean()[0])
#print("RMSE train: ", RMSE_train)
#print("Test RMSE: ", RMSE_test)

# Cross Validation
#val_pred = cross_val_predict(reg_mdl, chem_in_sort, chem_out, cv=5)
#RMSE_val = np.sqrt(metrics.mean_squared_error(chem_out, val_pred))
#print("Cross Validation RMSE: ", RMSE_val)


def reg_model(reg_mdl, x, y):
    # Split the data into training(70%) and testing(30%) sets
    chem_x_train, chem_x_test, chem_y_train, chem_y_test = \
        train_test_split(x, y, train_size=0.7, test_size=0.3)
    # Fit model using training data set
    reg_mdl = reg_mdl.fit(chem_x_train, chem_y_train)
    # Calculate output predictions for training and testing data sets
    pred_train = reg_mdl.predict(chem_x_train)
    pred_test = reg_mdl.predict(chem_x_test)
    # use KFold Cross Validation to check for generalization
    val_pred = cross_val_predict(reg_mdl, x, y, cv=5)
    # calculate RMSE of train, test and cross validation predictions
    RMSE_train = np.sqrt(metrics.mean_squared_error(chem_y_train, pred_train))
    RMSE_test = np.sqrt(metrics.mean_squared_error(chem_y_test, pred_test))
    RMSE_val = np.sqrt(metrics.mean_squared_error(chem_out, val_pred))
    # Return RMSE
    return [RMSE_train, RMSE_test, RMSE_val]


# forward selection to choose optimum number of variables for linear model
reg_mdl = LinearRegression()
RMSE_train = []
RMSE_test = []
RMSE_val = []

for i in range(1, len(chem_in_sort.columns) + 1):
    x = chem_in_sort.iloc[:, 0:i]
    y = chem_out.iloc[:, 0]
    RMSE = reg_model(reg_mdl, x, y)
    RMSE_train.append(RMSE[0])
    RMSE_test.append(RMSE[1])
    RMSE_val.append(RMSE[2])

# print the number of variables that gives the smallest validation RMSE
print(RMSE_val.index(min(RMSE_val)))
plt.plot(RMSE_train, "r", label="Training RMSE")
plt.plot(RMSE_test, "g", label="Testing RMSE")
plt.plot(RMSE_val, "b", label="Validation RMSE")
plt.xticks(range(1, len(chem_in_sort.columns) + 1))
plt.legend()
plt.show()
