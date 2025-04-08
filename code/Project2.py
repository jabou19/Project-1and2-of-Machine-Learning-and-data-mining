 #%%
from ucimlrepo import fetch_ucirepo 
import importlib_resources
import numpy as np
import pandas as pd
import torch
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


# %%
# Part B

# K-fold CrossValidation (4 folds here to speed up this example)
K = 10
CV = model_selection.KFold(K, shuffle=True)

# Setup figure for display of the decision boundary for the several crossvalidation folds.
decision_boundaries = plt.figure(1, figsize=(10, 10))
# Determine a size of a plot grid that fits visualizations for the chosen number
# of cross-validation splits, if K=4, this is simply a 2-by-2 grid.
subplot_size_1 = int(np.floor(np.sqrt(K)))
subplot_size_2 = int(np.ceil(K / subplot_size_1))
# Set overall title for all of the subplots
plt.suptitle("Data and model decision boundaries", fontsize=20)
# Change spacing of subplots
plt.subplots_adjust(left=0, bottom=0, right=1, top=0.9, wspace=0.5, hspace=0.25)

# Setup figure for display of learning curves and error rates in fold
summaries, summaries_axes = plt.subplots(1, 2, figsize=(10, 5))
# Make a list for storing assigned color of learning curve for up to K=10
color_list = [
    "tab:orange",
    "tab:green",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "tab:cyan",
    "tab:red",
    "tab:blue",
]

# Define the model structure
n_hidden_units =   10# number of hidden units in the signle hidden layer
# The lambda-syntax defines an anonymous function, which is used here to
# make it easy to make new networks within each cross validation fold
model = lambda: torch.nn.Sequential(
    torch.nn.Linear(M, n_hidden_units),  # M features to H hiden units
    # 1st transfer function, either Tanh or ReLU:
    torch.nn.Tanh(),  # torch.nn.ReLU(),
    torch.nn.Linear(n_hidden_units, 1),  # H hidden units to 1 output neuron
    torch.nn.Sigmoid(),  # final tranfer function
)
# Since we're training a neural network for binary classification, we use a
# binary cross entropy loss (see the help(train_neural_net) for more on
# the loss_fn input to the function)
loss_fn = torch.nn.CrossEntropyLoss()
# Train for a maximum of 10000 steps, or until convergence (see help for the
# function train_neural_net() for more on the tolerance/convergence))
max_iter = 10000
print("Training model of type:\n{}\n".format(str(model())))

# Do cross-validation:
errors = []  # make a list for storing generalizaition error in each loop
# Loop over each cross-validation split. The CV.split-method returns the
# indices to be used for training and testing in each split, and calling
# the enumerate-method with this simply returns this indices along with
# a counter k:
for k, (train_index, test_index) in enumerate(CV.split(X, y)):
    print("\nCrossvalidation fold: {0}/{1}".format(k + 1, K))

    # Extract training and test set for current CV fold,
    # and convert them to PyTorch tensors
    X_train = torch.Tensor(X[train_index, :])
    y_train = torch.Tensor(y[train_index])
    X_test = torch.Tensor(X[test_index, :])
    y_test = torch.Tensor(y[test_index])

    # Go to the file 'dtuimldmtools.py' in the Tools sub-folder of the toolbox
    # and see how the network is trained (search for 'def train_neural_net',
    # which is the place the function below is defined)
    net, final_loss, learning_curve = train_neural_net(
        model, loss_fn, X=X_train, y=y_train, n_replicates=3, max_iter=max_iter
    )

    print("\n\tBest loss: {}\n".format(final_loss))

    # Determine estimated class labels for test set
    y_sigmoid = net(X_test)  # activation of final note, i.e. prediction of network
    y_test_est = (y_sigmoid > 0.5).type(
        dtype=torch.uint8
    )  # threshold output of sigmoidal function
    y_test = y_test.type(dtype=torch.uint8)
    # Determine errors and error rate
    e = y_test_est != y_test
    error_rate = (sum(e).type(torch.float) / len(y_test)).data.numpy()
    errors.append(error_rate)  # store error rate for current CV fold

    # Make a subplot for current cross validation fold that displays the
    # decision boundary over the original data, "background color" corresponds
    # to the output of the sigmoidal transfer function (i.e. before threshold),
    # white areas are areas of uncertainty, and a deaper red/blue means
    # that the network "is more sure" of a given class.
    plt.figure(decision_boundaries.number)
    plt.subplot(subplot_size_1, subplot_size_2, k + 1)
    plt.title("CV fold {0}".format(k + 1), color=color_list[k])
    predict = lambda x: net(torch.tensor(x, dtype=torch.float)).data.numpy()
    visualize_decision_boundary(
        predict,
        X,
        y,  # provide data, along with function for prediction
        attributeNames,
        classNames,  # provide information on attribute and class names
        train=train_index,
        test=test_index,  # provide information on partioning
        show_legend=k == (K - 1),
    )  # only display legend for last plot

    # Display the learning curve for the best net in the current fold
    (h,) = summaries_axes[0].plot(learning_curve, color=color_list[k])
    h.set_label("CV fold {0}".format(k + 1))
    summaries_axes[0].set_xlabel("Iterations")
    summaries_axes[0].set_xlim((0, max_iter))
    summaries_axes[0].set_ylabel("Loss")
    summaries_axes[0].set_title("Learning curves")

# Display the error rate across folds
summaries_axes[1].bar(
    np.arange(1, K + 1), np.squeeze(np.asarray(errors)), color=color_list
)
summaries_axes[1].set_xlabel("Fold")
summaries_axes[1].set_xticks(np.arange(1, K + 1))
summaries_axes[1].set_ylabel("Error rate")
summaries_axes[1].set_title("Test misclassification rates")

# Show the plots
# plt.show(decision_boundaries.number) # try these lines if the following code fails (depends on package versions)
# plt.show(summaries.number)
plt.show()

# Display a diagram of the best network in last fold
print("Diagram of best neural net in last fold:")
weights = [net[i].weight.data.numpy().T for i in [0, 2]]
biases = [net[i].bias.data.numpy() for i in [0, 2]]
tf = [str(net[i]) for i in [1, 3]]
draw_neural_net(weights, biases, tf)

# Print the average classification error rate
print(
    "\nGeneralization error/average error rate: {0}%".format(
        round(100 * np.mean(errors), 4)
    )
)




