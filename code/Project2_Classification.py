#%%
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
from dtuimldmtools import draw_neural_net, train_neural_net
from dtuimldmtools import visualize_decision_boundary


# -----------------------------------------------------------
# Fetch dataset and view information
# -----------------------------------------------------------
data = pd.read_csv("AbaloneDataset.csv")
  # Ensure the CSV file is in your working directory


# -----------------------------------------------------------
# Reprocess the data for classification on "sex"
# -----------------------------------------------------------
# Assume the first column ('sex') is the target and the remaining columns are features.
# Remove rows where 'sex' is 'I' as this won't be part of the classification process.
data = data[data.iloc[:, 0] != 'I']

# Extract target and map 'F' to 1 and 'M' to 0
yy = data.iloc[:, 0].map({'F': 1, 'M': 0}).values.astype(np.float32)
# Remove the 'sex' column from features
X = data.iloc[:, 1:]                  # Use the remaining columns as features

# Extract attribute (feature) names
attributeNames = X.columns.tolist()

#------------------------------------
# Prepare target labels and map them to integers
# -----------------------------------------------------------
N, M = X.shape                        # Determine the number of samples (N) and features (M)
C = 2                                 # Define the number of classes

# -----------------------------------------------------------
# Convert DataFrame features to a NumPy matrix and standardize
# -----------------------------------------------------------

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


# %%
# Part B

# Parameters for neural network classifier
n_hidden_units = 4  # number of hidden units
n_replicates = 1  # number of networks trained in each k-fold
max_iter = 5000

# K-fold crossvalidation
K = 5  # only three folds to speed up this example
CV = model_selection.KFold(K, shuffle=True)

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
# Define the model
model = lambda: torch.nn.Sequential(
    torch.nn.Linear(M, n_hidden_units),  # M features to H hiden units
    torch.nn.Tanh(),  # 1st transfer function,
    torch.nn.Linear(n_hidden_units, 1),  # C logits
    torch.nn.Sigmoid(),  # final tranfer function
)
loss_fn = torch.nn.BCELoss()  # notice how this is now a mean-squared-error loss

print("Training model of type:\n\n{}\n".format(str(model())))
errors = []  # make a list for storing generalizaition error in each loop
for k, (train_index, test_index) in enumerate(CV.split(x_normalized, yy)):
    print("\nCrossvalidation fold: {0}/{1}".format(k + 1, K))

    # Extract training and test set for current CV fold, convert to tensors
    X_train = torch.Tensor(x_normalized[train_index, :])
    y_train = torch.Tensor(yy[train_index]).unsqueeze(1)
    X_test = torch.Tensor(x_normalized[test_index, :])
    y_test = torch.Tensor(yy[test_index]).unsqueeze(1)

    # Train the net on training data
    net, final_loss, learning_curve = train_neural_net(
        model,
        loss_fn,
        X=X_train,
        y=y_train,
        n_replicates=n_replicates,
        max_iter=max_iter,
    )

    print("\n\tBest loss: {}\n".format(final_loss))

    # Determine estimated class labels for test set
    y_sigmoid = net(X_test)
    y_test_est = (y_sigmoid > 0.5).type(dtype=torch.uint8)

    # Determine errors and errors
    y_test = y_test.type(dtype=torch.uint8)

    e = y_test_est != y_test
    error_rate = (sum(e).type(torch.float) / len(y_test)).data.numpy()
    errors.append(error_rate)  # store error rate for current CV fold

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

print("Diagram of best neural net in last fold:")
weights = [net[i].weight.data.numpy().T for i in [0, 2]]
biases = [net[i].bias.data.numpy() for i in [0, 2]]
tf = [str(net[i]) for i in [1, 3]]
draw_neural_net(weights, biases, tf, attribute_names=attributeNames)

# Print the average classification error rate
print(
    "\nGeneralization error/average error rate: {0}%".format(
        round(100 * np.mean(errors), 4)
    )
)

# %%
