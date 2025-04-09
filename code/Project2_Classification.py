#%%
import importlib_resources
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.linalg import svd
import sklearn.linear_model as lm
from sklearn.linear_model import LogisticRegression
from scipy.io import loadmat
from sklearn import model_selection
from sklearn.model_selection import train_test_split
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
# Remove rows where 'sex' is 'I' as this won't be part of the classification process.
data = data[data.iloc[:, 0] != 'I']

# Extract target and map 'F' to 1 and 'M' to 0
yy = data.iloc[:, 0].map({'F': 1, 'M': 0}).values.astype(np.float32)
# Remove the 'sex' column from features
X = data.iloc[:, 1:]                  # Use the remaining columns as features

# Extract attribute (feature) names
attributeNames = X.columns.tolist()

X = X.drop(columns=["Rings"])  # adjust the name if needed
X = X.drop(columns=["Height"])  # adjust the name if needed

#------------------------------------
# Prepare target labels and map them to integers
# -----------------------------------------------------------
N, M = X.shape                        # Determine the number of samples (N) and features (M)
C = 2                                 # Define the number of classes


# %%
k = 20
# Set the training and test set sizes for cross-validation
# Create cross-validation partition for evaluation using stratification and 90% split between training and test
X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(X, yy, test_size=.9, stratify=yy, shuffle=True)

# Standardize the training and set set based on training set mean and std
mu = np.mean(X_train_lr, 0)
sigma = np.std(X_train_lr, 0)

X_train_lr = (X_train_lr - mu) / sigma
X_test_lr = (X_test_lr - mu) / sigma

# Fit regularized logistic regression model to training data to predict sex of abalones
lambda_interval = np.logspace(-4, 4, 50)
train_error_rate_lr = np.zeros(len(lambda_interval))
test_error_rate_lr = np.zeros(len(lambda_interval))
coefficient_norm = np.zeros(len(lambda_interval))

for k in range(0, len(lambda_interval)):
    mdl = LogisticRegression(penalty="l2", C=1 / lambda_interval[k])

    mdl.fit(X_train_lr, y_train_lr)

    y_train_est_lr = mdl.predict(X_train_lr).T
    y_test_est_lr = mdl.predict(X_test_lr).T

    train_error_rate_lr[k] = np.sum(y_train_est_lr != y_train_lr) / len(y_train_lr)
    test_error_rate_lr[k] = np.sum(y_test_est_lr != y_test_lr) / len(y_test_lr)

    w_est = mdl.coef_[0]
    coefficient_norm[k] = np.sqrt(np.sum(w_est**2))

min_error = np.min(test_error_rate_lr)
opt_lambda_idx = np.argmin(test_error_rate_lr)
opt_lambda = lambda_interval[opt_lambda_idx]

plt.figure(figsize=(8, 8))
# plt.plot(np.log10(lambda_interval), train_error_rate*100)
# plt.plot(np.log10(lambda_interval), test_error_rate*100)
# plt.plot(np.log10(opt_lambda), min_error*100, 'o')
plt.semilogx(lambda_interval, train_error_rate_lr * 100)
plt.semilogx(lambda_interval, test_error_rate_lr * 100)
plt.semilogx(opt_lambda, min_error * 100, "o")
plt.text(
    1e-8,
    3,
    "Minimum test error: "
    + str(np.round(min_error * 100, 2))
    + " % at 1e"
    + str(np.round(np.log10(opt_lambda), 2)),
)
plt.xlabel("Regularization strength, $\log_{10}(\lambda)$")
plt.ylabel("Error rate (%)")
plt.title("Classification error")
plt.legend(["Training error", "Test error", "Test minimum"], loc="upper right")
plt.ylim([0, 100])
plt.grid()
plt.show()

plt.figure(figsize=(8, 8))
plt.semilogx(lambda_interval, coefficient_norm, "k")
plt.ylabel("L2 Norm")
plt.xlabel("Regularization strength, $\log_{10}(\lambda)$")
plt.title("Parameter vector L2 norm")
plt.grid()
plt.show()

# %%
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

# Get the shape of the normalized data (number of samples N and features M)
N, M = x_normalized.shape

# Parameters for neural network classifier
n_hidden_units = 5  # number of hidden units
n_replicates = 1  # number of networks trained in each k-fold
max_iter = 5000

# K-fold crossvalidation
K = 2  # only three folds to speed up this example
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
# -----------------------------------------------------------
# out and in Cross-Validation for Hyperparameter Tuning
# -----------------------------------------------------------

# Candidate number of hidden units to test in the in loop
candidate_hidden_units = [3, 4, 5, 6]

# out CV: 10 folds
K_out = 10
out_CV = model_selection.KFold(K_out, shuffle=True)

out_errors = []  # To store out fold test error
chosen_hidden_units = []  # To store chosen candidate per out fold

for out_fold, (out_train_idx, out_test_idx) in enumerate(out_CV.split(x_normalized, yy)):
    print(f"\nout CV fold {out_fold+1}/{K_out}")
    # Prepare out train and test data
    X_out_train = x_normalized[out_train_idx, :]
    y_out_train = yy[out_train_idx]
    X_out_test = x_normalized[out_test_idx, :]
    y_out_test = yy[out_test_idx]
    
    # in CV: use 10 folds on the out training set for model selection
    K_in = 10
    in_CV = model_selection.KFold(K_in, shuffle=True)
    
    # Store average in validation error for each candidate hidden unit
    candidate_avg_errors = []
    
    for h in candidate_hidden_units:
        in_errors = []  # Collect in validation error for current candidate
        for in_train_idx, in_val_idx in in_CV.split(X_out_train, y_out_train):
            # Create in training and validation sets
            X_in_train = torch.Tensor(X_out_train[in_train_idx, :])
            y_in_train = torch.Tensor(y_out_train[in_train_idx]).unsqueeze(1)
            X_in_val = torch.Tensor(X_out_train[in_val_idx, :])
            y_in_val = torch.Tensor(y_out_train[in_val_idx]).unsqueeze(1)
            
            # Define a model using candidate h hidden units and same loss function
            in_model = lambda: torch.nn.Sequential(
                torch.nn.Linear(M, h),      # Adjust input -> h hidden units
                torch.nn.Tanh(),
                torch.nn.Linear(h, 1),
                torch.nn.Sigmoid(),
            )
            
            # Train the network on the in training set
            net_in, final_loss_in, learning_curve_in = train_neural_net(
                in_model,
                loss_fn,
                X=X_in_train,
                y=y_in_train,
                n_replicates=n_replicates,
                max_iter=max_iter,
            )
            
            # Compute validation error on in validation set
            y_val_pred = net_in(X_in_val)
            y_val_est = (y_val_pred > 0.5).type(dtype=torch.uint8)
            error_in = (torch.sum(y_val_est != y_in_val.type(dtype=torch.uint8)).type(torch.float) 
                           / len(y_in_val)).data.numpy()
            in_errors.append(error_in)
        
        avg_in_error = np.mean(in_errors)
        candidate_avg_errors.append(avg_in_error)
        print(f"   Candidate hidden units: {h} with average in error: {avg_in_error*100:.2f}%")
        
    # Select best candidate (lowest average in validation error)
    best_idx = np.argmin(candidate_avg_errors)
    best_hidden = candidate_hidden_units[best_idx]
    chosen_hidden_units.append(best_hidden)
    print(f"Best candidate for out fold {out_fold+1}: {best_hidden} hidden units")
    
    # Retrain on the full out training set with the selected hyperparameter
    final_model = lambda: torch.nn.Sequential(
        torch.nn.Linear(M, best_hidden),
        torch.nn.Tanh(),
        torch.nn.Linear(best_hidden, 1),
        torch.nn.Sigmoid(),
    )
    
    X_out_train_tensor = torch.Tensor(X_out_train)
    y_out_train_tensor = torch.Tensor(y_out_train).unsqueeze(1)
    X_out_test_tensor = torch.Tensor(X_out_test)
    y_out_test_tensor = torch.Tensor(y_out_test).unsqueeze(1)
    
    net_final, final_loss_final, learning_curve_final = train_neural_net(
        final_model,
        loss_fn,
        X=X_out_train_tensor,
        y=y_out_train_tensor,
        n_replicates=n_replicates,
        max_iter=max_iter,
    )
    
    # Evaluate on out test set
    y_out_pred = net_final(X_out_test_tensor)
    y_out_est = (y_out_pred > 0.5).type(dtype=torch.uint8)
    out_error = (torch.sum(y_out_est != y_out_test_tensor.type(dtype=torch.uint8)).type(torch.float) 
                   / len(y_out_test_tensor)).data.numpy()
    out_errors.append(out_error)
    print(f"out fold {out_fold+1} test error: {out_error*100:.2f}%")
    
# Report overall performance
avg_out_error = np.mean(out_errors)
print(f"\nOverall generalization error: {avg_out_error*100:.2f}%")
print("Chosen hidden units per out fold:", chosen_hidden_units)

# %%
