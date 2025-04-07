 #%%
from ucimlrepo import fetch_ucirepo 
import importlib_resources
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import svd
import sklearn.linear_model as lm
from scipy.io import loadmat
from sklearn import model_selection
from dtuimldmtools import rlr_validate
  
# -----------------------------------------------------------
# Fetch dataset and view information
# -----------------------------------------------------------
abalone = fetch_ucirepo(id=1)  # Fetch dataset using ucimlrepo
X = abalone.data.features    # Extract features as a pandas DataFrame
y = abalone.data.targets     # Extract targets as a pandas DataFrame

raw_data = X.values         # Convert DataFrame to a raw NumPy array
print(abalone.metadata)     # Print dataset metadata
print(abalone.variables)    # Print variable information

# -----------------------------------------------------------
# Preprocess the data
# -----------------------------------------------------------
cols = range(0, 10)  # Use the first 10 columns for attribute names

# Convert 'sex' column into three binary attributes: Male, Female, Infant
X['Male'] = (X.iloc[:, 0] == 'M').astype(int)
X['Female'] = (X.iloc[:, 0] == 'F').astype(int)
X['Infant'] = (X.iloc[:, 0] == 'I').astype(int)

# Drop the original 'sex' column since it's now encoded into binary features
X = X.drop(columns=[X.columns[0]])

# Extract attribute names from the specified columns
attributeNames = np.asarray(X.columns[cols]).tolist()

# -----------------------------------------------------------
# Prepare target labels and map them to integers
# -----------------------------------------------------------
classLabels = y.values[:, 0]          # Extract class labels from the target DataFrame
classNames = np.unique(classLabels).tolist()  # Get the unique class names
classDict = dict(zip(classNames, range(1, len(classNames) + 1)))  # Map classes to integer values

N, M = X.shape  # Determine the number of samples (N) and features (M)
C = len(classNames)  # Determine the number of classes

# -----------------------------------------------------------
# Convert DataFrame features to a NumPy matrix and standardize
# -----------------------------------------------------------

# Convert textual class labels to numerical values using the mapping dictionary
yy = np.array([classDict[value] for value in classLabels])

# Preallocate memory for the feature matrix and populate with data from DataFrame X
xx = np.empty((N, M))
for i in range(M):
    xx[:, i] = np.array(X.iloc[:, i]).T

# Standardize data by subtracting the mean and dividing by the standard deviation for each feature
x_normalized = np.empty((N, M))
for i in range(M):
    x_normalized[:, i] = (xx[:, i] - np.mean(xx[:, i])) / np.std(xx[:, i])

#%%
#PROJECT 2
# -----------------------------------------------------------------------------
# Regularization factor
# -----------------------------------------------------------------------------
# Get the shape of the normalized data (number of samples N and features M)
N, M = x_normalized.shape

# -----------------------------------------------------------------------------
# Add offset attribute
# -----------------------------------------------------------------------------
# Add a column of ones to the feature matrix to account for the bias term
x_normalized = np.concatenate((np.ones((x_normalized.shape[0], 1)), x_normalized), 1)

# Update the attribute names to include the "Offset" feature
attributeNames = ["Offset"] + attributeNames

# Increment the feature count to reflect the added offset
M = M + 1

# -----------------------------------------------------------------------------
# Cross-validation setup
# -----------------------------------------------------------------------------
# Define the number of folds for cross-validation
K = 10

# Create a KFold cross-validation object with shuffling enabled
CV = model_selection.KFold(K, shuffle=True)

# Alternative: Uncomment the following line to disable shuffling
# CV = model_selection.KFold(K, shuffle=False)

# -----------------------------------------------------------------------------
# Define regularization parameters
# -----------------------------------------------------------------------------
# Generate a range of lambda values (regularization factors) as powers of 10
lambdas = np.power(10.0, range(-5, 9))

# -----------------------------------------------------------------------------
# Initialize variables for storing results
# -----------------------------------------------------------------------------
# Arrays to store training and test errors for different models
Error_train = np.empty((K, 1))             # Training error for unregularized model
Error_test = np.empty((K, 1))              # Test error for unregularized model
Error_train_rlr = np.empty((K, 1))         # Training error for regularized model
Error_test_rlr = np.empty((K, 1))          # Test error for regularized model
Error_train_nofeatures = np.empty((K, 1))  # Baseline training error (no features)
Error_test_nofeatures = np.empty((K, 1))   # Baseline test error (no features)

# Arrays to store model weights for each fold
w_rlr = np.empty((M, K))  # Weights for regularized linear regression
w_noreg = np.empty((M, K))  # Weights for unregularized linear regression

# Arrays to store mean and standard deviation for standardizing features
mu = np.empty((K, M - 1))     # Mean of features (excluding offset) for each fold
sigma = np.empty((K, M - 1))  # Standard deviation of features (excluding offset)


k = 0
for train_index, test_index in CV.split(x_normalized, yy):
    # Extract training and test sets for the current cross-validation fold
    X_train = x_normalized[train_index]
    y_train = yy[train_index]
    X_test = x_normalized[test_index]
    y_test = yy[test_index]
    internal_cross_validation = 10  # Number of folds for internal cross-validation

    # Perform regularized linear regression validation to find the optimal lambda
    (
        opt_val_err,          # Optimal validation error
        opt_lambda,           # Optimal lambda value
        mean_w_vs_lambda,     # Mean weights for different lambda values
        train_err_vs_lambda,  # Training error vs. lambda
        test_err_vs_lambda    # Test error vs. lambda
    ) = rlr_validate(X_train, y_train, lambdas, internal_cross_validation)

    # Standardize the outer fold based on the training set
    # Save the mean and standard deviation for future predictions
    mu[k, :] = np.mean(X_train[:, 1:], axis=0)  # Compute mean (excluding offset)
    sigma[k, :] = np.std(X_train[:, 1:], axis=0)  # Compute standard deviation (excluding offset)

    # Apply standardization to the training and test sets (excluding the offset)
    X_train[:, 1:] = (X_train[:, 1:] - mu[k, :]) / sigma[k, :]
    X_test[:, 1:] = (X_test[:, 1:] - mu[k, :]) / sigma[k, :]

   # Compute the feature-target product and feature cross-product for weight estimation
    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train

    # Compute baseline mean squared error without using any features
    Error_train_nofeatures[k] = (
        np.square(y_train - y_train.mean()).sum(axis=0) / y_train.shape[0]
    )
    Error_test_nofeatures[k] = (
        np.square(y_test - y_test.mean()).sum(axis=0) / y_test.shape[0]
    )

    # Estimate weights for the optimal lambda using regularized linear regression
    lambdaI = opt_lambda * np.eye(M)  # Create regularization matrix
    lambdaI[0, 0] = 0  # Do not regularize the bias term
    w_rlr[:, k] = np.linalg.solve(XtX + lambdaI, Xty).squeeze()  # Solve for weights

    # Compute mean squared error for the regularized model
    Error_train_rlr[k] = (
        np.square(y_train - X_train @ w_rlr[:, k]).sum(axis=0) / y_train.shape[0]
    )
    Error_test_rlr[k] = (
        np.square(y_test - X_test @ w_rlr[:, k]).sum(axis=0) / y_test.shape[0]
    )

    # Estimate weights for unregularized linear regression
    w_noreg[:, k] = np.linalg.solve(XtX, Xty).squeeze()  # Solve for weights

    # Compute mean squared error for the unregularized model
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
    print('Cross validation fold {0}/{1}:'.format(k+1,K))
    print('Train indices: {0}'.format(train_index))
    print('Test indices: {0}\n'.format(test_index))

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


