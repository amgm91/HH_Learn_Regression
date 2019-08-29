import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

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

for i in range(len(chem_in.columns)):
    plt.scatter(chem_in_std[i], chem_out, label="variable %d and output" %i)
    plt.xlabel(("var %d" % i))
    plt.ylabel("output")
    plt.show()



