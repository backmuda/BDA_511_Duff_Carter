"""
Zach Duff: December 2018
"""

import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.mlab import PCA


train_file = "train_data.csv"
test_file = "test_data.csv"

# pandas dataframes
df_train = pd.read_csv(train_file, header=0)
df_test = pd.read_csv(test_file, header=0)

# Dependent Variables
y_train = df_train.SalePrice.values #pd --> np
y_train = y_train.reshape(len(y_train), 1)
avg_train = y_train.mean()
print("Number of Training Samples: %s" % len(y_train))
print("Average Training House Price: " + "${:,.2f}".format(avg_train) + '\n')


y_test = df_test.SalePrice.values #pd --> np
y_test = y_test.reshape(len(y_test), 1)
avg_test = y_test.mean()
print("Number of Testing Samples: %s" % len(y_test))
print("Average Training House Price: " + "${:,.2f}".format(avg_test) + '\n') #This one just for fun.

# Error
E0 = sum([(avg_train - y) ** 2 for y in y_test])

#------------- Data Cleansing --------------------

# Get Numeric Features (not necessarily useful)
df_train_num = df_train.select_dtypes(include=['float64', 'int64'])

df_train_num = df_train_num.drop(labels=['MSSubClass', 'Id', 'SalePrice'], axis=1)
df_train_num = df_train_num.dropna()


# --------------- Data Exploration ----------------

corr = df_train_num.corr()
fig = plt.figure()
fig.set_size_inches([8., 6.])
ax = fig.add_subplot(111)
cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(df_train_num.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(df_train_num.columns)
ax.set_yticklabels(df_train_num.columns)
plt.show()

max_list = {}
for column_name in corr.columns:
    m = float(-1)
    for corr_coeff in corr[column_name]:
        if corr_coeff < 1:
            if corr_coeff < 0:
                m = abs(m)
                a_coeff = abs(corr_coeff)
                m = max(m, a_coeff)
                m = -(m)
            else:
                m = max(m, corr_coeff)
    max_list[column_name] = m
    
c = corr.abs().unstack()
c = c.sort_values(kind="quicksort")
    
    


