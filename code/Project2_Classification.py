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
from dtuimldmtools import draw_neural_net, train_neural_net
from dtuimldmtools import visualize_decision_boundary


# -----------------------------------------------------------
# Fetch dataset and view information
# -----------------------------------------------------------
data = pd.read_csv("C:\Workspace\Git\Master repos\Project2-Machine-Learning-and-data-mining\code\AbaloneDataset.csv")
  # Ensure the CSV file is in your working directory


# -----------------------------------------------------------
# Reprocess the data for classification on "sex"
# -----------------------------------------------------------
# Assume the first column ('sex') is the target and the remaining columns are features.
# Extract target
y = data.iloc[:, 0].map({'M': -1, 'F': 1, 'I': 0})  # Map 'M', 'F', 'I' to -1, 1, 0 as targets

# Remove the 'sex' column from features
X = data.iloc[:, 1:]                  # Use the remaining columns as features

# Extract attribute (feature) names
attributeNames = X.columns.tolist()

# Convert target labels (e.g., 'M', 'F', 'I') to numeric values
classNames = np.unique(y).tolist()    # List unique class names
classDict = {label: idx for idx, label in enumerate(classNames)}  # Map each class to an integer
yy = y.map(classDict).values          # Convert target labels to numerical values

# Convert the features from DataFrame to a NumPy array
raw_data = X.values

# -----------------------------------------------------------
# Prepare target labels and map them to integers
# -----------------------------------------------------------
classLabels = y.values                # Extract class labels from the target DataFrame
N, M = X.shape                        # Determine the number of samples (N) and features (M)
C = len(classNames)                   # Determine the number of classes

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

# %%
# Part B

# Parameters for neural network classifier
n_hidden_units = 5  # number of hidden units
n_replicates = 1  # number of networks trained in each k-fold
max_iter = 5000

# K-fold crossvalidation
K = 4  # only three folds to speed up this example
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
    torch.nn.ReLU(),  # 1st transfer function
    # Output layer:
    # H hidden units to C classes
    # the nodes and their activation before the transfer
    # function is often referred to as logits/logit output
    torch.nn.Linear(n_hidden_units, C),  # C logits
    # To obtain normalised "probabilities" of each class
    # we use the softmax-funtion along the "class" dimension
    # (i.e. not the dimension describing observations)
    torch.nn.Softmax(dim=1),  # final tranfer function, normalisation of logit output
)
loss_fn = torch.nn.CrossEntropyLoss()  # notice how this is now a mean-squared-error loss

print("Training model of type:\n\n{}\n".format(str(model())))
errors = []  # make a list for storing generalizaition error in each loop
for k, (train_index, test_index) in enumerate(CV.split(x_normalized, yy)):
    print("\nCrossvalidation fold: {0}/{1}".format(k + 1, K))

    # Extract training and test set for current CV fold, convert to tensors
    X_train = torch.Tensor(x_normalized[train_index, :])
    y_train = torch.LongTensor(yy[train_index])
    X_test = torch.Tensor(x_normalized[test_index, :])
    y_test = torch.LongTensor(yy[test_index])

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

   # Determine probability of each class using trained network
    softmax_logits = net(torch.tensor(X_test, dtype=torch.float))
    # Get the estimated class as the class with highest probability (argmax on softmax_logits)
    y_test_est = (torch.max(softmax_logits, dim=1)[1]).data.numpy()
    # Determine errors
    e = y_test_est != y_test
    print(
        "Number of miss-classifications for ANN:\n\t {0} out of {1}".format(sum(e), len(e))
    )

    predict = lambda x: (
    torch.max(net(torch.tensor(x, dtype=torch.float)), dim=1)[1]
        ).data.numpy()
    plt.figure(1, figsize=(9, 9))
    visualize_decision_boundary(
        predict, [X_train, X_test], [y_train, y_test], attributeNames, classNames
    )
    plt.title("ANN decision boundaries")




# Print the number of misclassifications
print(
    "Number of miss-classifications for ANN:\n\t {0} out of {1}".format(
        sum(e), len(e)
    )
)

print("Ran Exercise 8.2.5")





# %%
