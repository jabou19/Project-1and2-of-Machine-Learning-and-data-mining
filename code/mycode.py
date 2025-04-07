# %%
# import matplotlib
# matplotlib.use('TkAgg')  # Try 'Qt5Agg' if 'TkAgg' doesn't work
from ucimlrepo import fetch_ucirepo
import importlib_resources
import numpy as np
import pandas as pd

# fetch dataset
abalone = fetch_ucirepo(id=1)
# Check the keys in the dataset
print("Keys in the dataset:", abalone.keys())
# data (as pandas dataframes) fafa
X = abalone.data.features
y = abalone.data.targets
raw_data = X.values
# metadata
print("Metadata:", abalone.meta)

# variable information
print("Variable information:\n", abalone.variables)

cols = range(0, 8)
attributeNames = np.asarray(X.columns[cols]).tolist()
print("Attribute names:\n", attributeNames)
# values[:,0] means that we take the first column of the values
classLabels = y.values[:, -1]
print("Class labels:\n", classLabels)
classNames = np.unique(classLabels).tolist()
print("Class names:\n", classNames)
classDict = dict(zip(classNames, range(len(classNames))))
print("Class dictionary:\n", classDict)

N, M = X.shape
C = len(classNames)
print("Data Loaded: {0} samples or rows, {1} attributes, {2} classes".format(N, M, C))

# Extract vector y, convert to NumPy matrix and transpose
yy = np.array([classDict[value] for value in classLabels])
# print("yy shape:", yy.shape)

# Preallocate memory, then extract data to matrix X
xx = np.empty((N, M))
for i in range(M):
    if i != 0:
        # X.iloc[:,i] icloc is used to access a group of rows and columns, her i is to access the ith column.
        xx[:, i] = np.array(X.iloc[:, i]).T  # T is the transpose function
    else:
        for j in range(N):
            # X.iloc[j, i] is referring to the jth row and ith column of the data
            if X.iloc[j, i] == 'M':  # sex = 0 for males
                xx[j, i] = 0
            elif X.iloc[j, i] == 'F':  # sex = 1 for females
                xx[j, i] = 1
            else:
                xx[j, i] = 2  # sex = 2 for infants الأطفال الرضع

# %%
"" "Exercise 2.3.2"""
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
u = np.floor(np.sqrt(M))  # u is the number of rows, floor is used to round down to the nearest integer
v = np.ceil(float(M) / u)  # v is the number of columns, ceil is used to round up to the nearest integer
for i in range(M):
    plt.subplot(int(u), int(v), i + 1)  # subplot(nrows, ncols, index)
    plt.hist(xx[:, i],
             color=(0.2, 0.8 - i / 4 * 0.2, 0.4))  # hist(xx[:, i]) where is xx[:, i] is the ith column of the data
    plt.xlabel(attributeNames[i])
    plt.ylim(0, N / 2)  # ylimit is used to set the range of the y-axis
    # Adjust vertical spacing
    plt.subplots_adjust(hspace=0.2, wspace=0.5)  # Increase spacing between subplots
# plt.show()

# %%
""" standardize the data"""
# Subtract the mean from the data
# Z-score normalization formula:  X_normalized𝑖 = (X𝑖 - mean(X𝑖)) / std(X𝑖)
# X𝑖 is the original feature value (column 𝑖)., mean(X𝑖) is the mean of the feature values, std(X𝑖) is the standard deviation of the feature values
x_normalized = np.empty((N, M))
for i in range(M):
    x_normalized[:, i] = (xx[:, i] - np.mean(xx[:, i])) / np.std(xx[:, i])

# %%
""" Exercise 2.3.3"""
"""  a boxplot of the attributes of standardized Abalone data"""
plt.figure()
# plt.boxplot(xx)   ???????????????????????????
plt.boxplot(x_normalized)
plt.xticks(range(1, 9), attributeNames, rotation=16)
# plt.ylabel("cm")
plt.title("Standardized Abalone boxplot")
# plt.show()


# %%

f, ax = plt.subplots(4, C // 4, figsize=(14, 40))

for c in range(C):
    class_mask = yy == c  # binary mask to extract elements of class c
    # or: class_mask = nonzero(y==c)[0].tolist()[0] # indices of class c

    ax.ravel()[c].boxplot(x_normalized[class_mask, :])
    # title('Class: {0}'.format(classNames[c]))
    ax.ravel()[c].set_title("Class: " + str(classNames[c]))
    # ax.ravel()[c].xticks(
    #     range(1, len(attributeNames) + 1), [a[:7] for a in attributeNames], rotation=45
    # )
    ax.ravel()[c].set_xticklabels(attributeNames, rotation=90)  # Rotate x-ticks
    # y_up = x_normalized.max() + (x_normalized.max() - x_normalized.min()) * 0.1
    # y_down = x_normalized.min() - (x_normalized.max() - x_normalized.min()) * 0.1
    # ax.ravel()[c].set_ylim(y_down, y_up)

# plt.show()

# %%
"""Exercise 2.3.5"""
""" matrix of scatter plots of each combination of two attributes against each other"""
plt.figure(figsize=(12, 10))
# two nested loops to iterate over all combinations of attributes
for m1 in range(M):
    for m2 in range(M):
        # subplot(rows, columns, index), start plotting from top left
        plt.subplot(M, M, m1 * M + m2 + 1)
        # C is the number of classes
        for c in range(C):
            class_mask = yy == c  # yy is the class labels
            # plot(x, y, marker) is used to plot the data
            plt.plot(np.array(x_normalized[class_mask, m2]), np.array(x_normalized[class_mask, m1]), ".")
            if m1 == M - 1:
                plt.xlabel(attributeNames[m2])
            else:
                plt.xticks([])
            if m2 == 0:
                plt.ylabel(attributeNames[m1])
            else:
                plt.yticks([])

# plt.legend(classNames)
# bbox_to_anchor(x, y) where x and y are the coordinates of the legend
plt.legend(classNames, loc='center left', bbox_to_anchor=(1.2, 5), title="Classes")
plt.tight_layout()  # Adjust layout to prevent overlapping
plt.show()
# %%

