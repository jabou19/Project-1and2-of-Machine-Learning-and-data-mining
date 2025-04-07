#%%
from ucimlrepo import fetch_ucirepo 
import importlib_resources
import numpy as np
import pandas as pd
  
# fetch dataset 
abalone = fetch_ucirepo(id=1) 
# data (as pandas dataframes) fafa
X = abalone.data.features 
y = abalone.data.targets 

raw_data = X.values
# metadata 
print(abalone.metadata) 
  
# variable information 
print(abalone.variables) 

cols = range(0, 10)
# Convert 'sex' column into three binary attributes: Male, Female, Infant
X['Male'] = (X.iloc[:, 0] == 'M').astype(int)
X['Female'] = (X.iloc[:, 0] == 'F').astype(int)
X['Infant'] = (X.iloc[:, 0] == 'I').astype(int)

# Drop the original 'sex' column
X = X.drop(columns=[X.columns[0]])

attributeNames = np.asarray(X.columns[cols]).tolist()

classLabels = y.values[:, 0]
classNames = np.unique(classLabels).tolist()


classDict = dict(zip(classNames, range(1,len(classNames)+1)))


N, M = X.shape

C = len(classNames)

#%%

# Extract vector y, convert to NumPy matrix and transpose
yy = np.array([classDict[value] for value in classLabels])

# Preallocate memory, then extract data to matrix X
xx = np.empty((N, M))
for i in range(M):

    xx[:, i] = np.array(X.iloc[:,i]).T



# %%
import matplotlib.pyplot as plt
f = plt.figure(figsize=(16, 10))
f.suptitle("Abalone barplot", fontsize=24)  # Add a title to the entire figure
u = np.floor(np.sqrt(M))
v = np.ceil(float(M) / u)
for i in range(M):
    plt.subplot(int(u), int(v), i + 1)
    plt.hist(xx[:, i], color=(0.2, 0.8 - i/4 * 0.2, 0.4))
    plt.xlabel(attributeNames[i])
    plt.ylim(0, N / 1.3)
plt.subplot(int(u), int(v), 11)
plt.hist(yy[:], color=(0.2, 0.8 - i/4 * 0.2, 0.4))
plt.xlabel("Class Labels")

plt.show()
# %% Standardize the data
# Subtract the mean from the data   

x_normalized = np.empty((N, M))
for i in range(M):
    x_normalized[:, i] = (xx[:, i] - np.mean(xx[:, i]))/np.std(xx[:, i])


# %%
plt.figure()
plt.boxplot(x_normalized[:,:7])
plt.xticks(range(1, 8), attributeNames[:7], rotation=60)
#plt.ylabel("cm")
plt.title("Standardized Abalone boxplot")
plt.show()

print("Ran Exercise 2.3.3")


# %%

f, ax = plt.subplots(4,C//4, figsize=(14, 40))

for c in range(C):
    
    class_mask = yy == c # binary mask to extract elements of class c
    # or: class_mask = nonzero(y==c)[0].tolist()[0] # indices of class c

    ax.ravel()[c].boxplot(x_normalized[class_mask, :])
    # title('Class: {0}'.format(classNames[c]))
    ax.ravel()[c].set_title("Class: " + str(classNames[c]))
    # ax.ravel()[c].xticks(
    #     range(1, len(attributeNames) + 1), [a[:7] for a in attributeNames], rotation=45
    # )
    ax.ravel()[c].set_xticklabels(attributeNames, rotation=90)  # Rotate x-ticks
    #y_up = x_normalized.max() + (x_normalized.max() - x_normalized.min()) * 0.1
    #y_down = x_normalized.min() - (x_normalized.max() - x_normalized.min()) * 0.1
    #ax.ravel()[c].set_ylim(y_down, y_up)

plt.show()

# %%
plt.figure(figsize=(12, 10))
# Create a colormap
cmap = plt.get_cmap("Reds")

# Normalize the class labels for color mapping
norm = plt.Normalize(vmin=1, vmax=C)

# Create a grid of subplots
fig, axes = plt.subplots(M-3, M-3, figsize=(12, 10))

for m1 in range(M-3):
    for m2 in range(M-3):
        ax = axes[m1, m2]
        sc = ax.scatter(x_normalized[:, m2], x_normalized[:, m1], c=yy, cmap=cmap, norm=norm, alpha=0.5)
        if m1 == M - 4:
            ax.set_xlabel(attributeNames[m2])
        else:
            ax.set_xticks([])
        if m2 == 0:
            ax.set_ylabel(attributeNames[m1])
        else:
            ax.set_yticks([])

# Add a single colorbar for the entire figure
fig.colorbar(sc, ax=axes, label='Class Labels', orientation='vertical')

plt.show()






# %% PCA
from scipy.linalg import svd 

# Subtract mean value from data
# Note: Here we use Y to in teh book we often use X with a hat-symbol on top.
#Y = x_normalized[:,1:] - np.ones((N, 1)) * x_normalized[:,1:].mean(axis=0)
Y = x_normalized[:,:]

# PCA by computing SVD of Y
# Note: Here we call the Sigma matrix in the SVD S for notational convinience
U, S, Vh = svd(Y, full_matrices=False)
print(Vh)
# scipy.linalg.svd returns "Vh", which is the Hermitian (transpose)
# of the vector V. So, for us to obtain the correct V, we transpose:
V = Vh.T

# Compute variance explained by principal components 
# Note: This is an important equation, see Eq. 3.18 on page 40 in the book.
rho = (S * S) / (S * S).sum()

threshold = 0.9

# Plot variance explained
plt.figure()
plt.plot(range(1, len(rho) + 1), rho, "x-")
plt.plot(range(1, len(rho) + 1), np.cumsum(rho), "o-")
plt.plot([1, len(rho)], [threshold, threshold], "k--")
plt.title("Variance explained by principal components")
plt.xlabel("Principal component")
plt.ylabel("Variance explained")
plt.legend(["Individual", "Cumulative", "Threshold"])
plt.grid()
plt.show()

#%%

import matplotlib.pyplot as plt

# Data attributes to be plotted
i = 0
j = 1

##
# Make a simple plot of the i'th attribute against the j'th attribute
plt.plot(x_normalized[:, i], x_normalized[:, j], "o")

##
# Make another more fancy plot that includes legend, class labels,
# attribute names, and a title.
f = plt.figure()
plt.title("NanoNose data")

for c in range(C):
    # select indices belonging to class c:
    class_mask = yy == c 
    plt.plot(x_normalized[class_mask, i], x_normalized[class_mask, j], "o", alpha=0.3)

plt.legend(classNames)
plt.xlabel(attributeNames[i])
plt.ylabel(attributeNames[j])

# Output result to screen
plt.show()



# %%
# Project the centered data onto principal component space
# Note: Make absolutely sure you understand what the @ symbol 
# does by inspecting the numpy documentation!
Z = Y @ V

# Indices of the principal components to be plotted
i = 0
j = 1

# Plot PCA of the data
f = plt.figure()
plt.title("Abalone data: PCA")
# Create a colormap
cmap = plt.get_cmap("Reds")

# Normalize the class labels for color mapping
norm = plt.Normalize(vmin=1, vmax=C)

# Z = array(Z)
sc = plt.scatter(Z[:, i], Z[:, j], c=yy, cmap=cmap, norm=norm, alpha=0.5)
plt.colorbar(sc, label='Class Labels')  # Add colorbar

plt.xlabel("PC{0}".format(i + 1))
plt.ylabel("PC{0}".format(j + 1))

# Set y-axis limit
plt    # %%



#%%
#PROJECT 2
#regularization factor

# exercise 8.1.1

import importlib_resources
import numpy as np
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn import model_selection

from dtuimldmtools import rlr_validate


#X = mat_data["X"]
#y = mat_data["y"].squeeze()

N, M = x_normalized.shape

# Add offset attribute
x_normalized = np.concatenate((np.ones((x_normalized.shape[0], 1)), x_normalized), 1)
attributeNames = ["Offset"] + attributeNames
M = M + 1

## Crossvalidation
# Create crossvalidation partition for evaluation
K = 10
CV = model_selection.KFold(K, shuffle=True)
# CV = model_selection.KFold(K, shuffle=False)

# Values of lambda
lambdas = np.power(10.0, range(-5, 9))

# Initialize variables
# T = len(lambdas)
Error_train = np.empty((K, 1))
Error_test = np.empty((K, 1))
Error_train_rlr = np.empty((K, 1))
Error_test_rlr = np.empty((K, 1))
Error_train_nofeatures = np.empty((K, 1))
Error_test_nofeatures = np.empty((K, 1))
w_rlr = np.empty((M, K))
mu = np.empty((K, M - 1))
sigma = np.empty((K, M - 1))
w_noreg = np.empty((M, K))
#%%
k = 0
for train_index, test_index in CV.split(x_normalized, yy):
    # extract training and test set for current CV fold
    X_train = x_normalized[train_index]
    y_train = yy[train_index]
    X_test = x_normalized[test_index]
    y_test = yy[test_index]
    internal_cross_validation = 10

    (
        opt_val_err,
        opt_lambda,
        mean_w_vs_lambda,
        train_err_vs_lambda,
        test_err_vs_lambda,
    ) = rlr_validate(X_train, y_train, lambdas, internal_cross_validation)

    # Standardize outer fold based on training set, and save the mean and standard
    # deviations since they're part of the model (they would be needed for
    # making new predictions) - for brevity we won't always store these in the scripts
    mu[k, :] = np.mean(X_train[:, 1:], 0)
    sigma[k, :] = np.std(X_train[:, 1:], 0)

    X_train[:, 1:] = (X_train[:, 1:] - mu[k, :]) / sigma[k, :]
    X_test[:, 1:] = (X_test[:, 1:] - mu[k, :]) / sigma[k, :]

    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train

    # Compute mean squared error without using the input data at all
    Error_train_nofeatures[k] = (
        np.square(y_train - y_train.mean()).sum(axis=0) / y_train.shape[0]
    )
    Error_test_nofeatures[k] = (
        np.square(y_test - y_test.mean()).sum(axis=0) / y_test.shape[0]
    )

    # Estimate weights for the optimal value of lambda, on entire training set
    lambdaI = opt_lambda * np.eye(M)
    lambdaI[0, 0] = 0  # Do no regularize the bias term
    w_rlr[:, k] = np.linalg.solve(XtX + lambdaI, Xty).squeeze()
    # Compute mean squared error with regularization with optimal lambda
    Error_train_rlr[k] = (
        np.square(y_train - X_train @ w_rlr[:, k]).sum(axis=0) / y_train.shape[0]
    )
    Error_test_rlr[k] = (
        np.square(y_test - X_test @ w_rlr[:, k]).sum(axis=0) / y_test.shape[0]
    )

    # Estimate weights for unregularized linear regression, on entire training set
    w_noreg[:, k] = np.linalg.solve(XtX, Xty).squeeze()
    # Compute mean squared error without regularization
    Error_train[k] = (
        np.square(y_train - X_train @ w_noreg[:, k]).sum(axis=0) / y_train.shape[0]
    )
    Error_test[k] = (
        np.square(y_test - X_test @ w_noreg[:, k]).sum(axis=0) / y_test.shape[0]
    )
    # OR ALTERNATIVELY: you can use sklearn.linear_model module for linear regression:
    # m = lm.LinearRegression().fit(X_train, y_train)
    # Error_train[k] = np.square(y_train-m.predict(X_train)).sum()/y_train.shape[0]
    # Error_test[k] = np.square(y_test-m.predict(X_test)).sum()/y_test.shape[0]

    # Display the results for the last cross-validation fold
    if k == K - 1:
        plt.figure(k, figsize=(12, 8))
        plt.subplot(1, 2, 1)
        plt.semilogx(lambdas, mean_w_vs_lambda.T[:, 1:], ".-")  # Don't plot the bias term
        plt.xlabel("Regularization factor")
        plt.ylabel("Mean Coefficient Values")
        plt.grid()
        # You can choose to display the legend, but it's omitted for a cleaner
        # plot, since there are many attributes
        # legend(attributeNames[1:], loc='best')

        plt.subplot(1, 2, 2)
        plt.title("Optimal lambda: 1e{0}".format(np.log10(opt_lambda)))
        plt.loglog(
            lambdas, train_err_vs_lambda.T, "b.-", lambdas, test_err_vs_lambda.T, "r.-"
        )
        plt.xlabel("Regularization factor")
        plt.ylabel("Squared error (crossvalidation)")
        plt.legend(["Train error", "Validation error"])
        plt.grid()

    # To inspect the used indices, use these print statements
    # print('Cross validation fold {0}/{1}:'.format(k+1,K))
    # print('Train indices: {0}'.format(train_index))
    # print('Test indices: {0}\n'.format(test_index))

    k += 1

plt.show()
# Display results
print("Linear regression without feature selection:")
print("- Training error: {0}".format(Error_train.mean()))
print("- Test error:     {0}".format(Error_test.mean()))
print(
    "- R^2 train:     {0}".format(
        (Error_train_nofeatures.sum() - Error_train.sum())
        / Error_train_nofeatures.sum()
    )
)
print(
    "- R^2 test:     {0}\n".format(
        (Error_test_nofeatures.sum() - Error_test.sum()) / Error_test_nofeatures.sum()
    )
)
print("Regularized linear regression:")
print("- Training error: {0}".format(Error_train_rlr.mean()))
print("- Test error:     {0}".format(Error_test_rlr.mean()))
print(
    "- R^2 train:     {0}".format(
        (Error_train_nofeatures.sum() - Error_train_rlr.sum())
        / Error_train_nofeatures.sum()
    )
)
print(
    "- R^2 test:     {0}\n".format(
        (Error_test_nofeatures.sum() - Error_test_rlr.sum())
        / Error_test_nofeatures.sum()
    )
)

print("Weights in last fold:")
for m in range(M):
    print("{:>15} {:>15}".format(attributeNames[m], np.round(w_rlr[m, -1], 2)))

print("Ran Exercise 8.1.1")

# %%


