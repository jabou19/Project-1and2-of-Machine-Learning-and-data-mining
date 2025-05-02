#%%
# --- Imports and Setup ---
from scipy import stats
from ucimlrepo import fetch_ucirepo
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn import model_selection
from dtuimldmtools import rlr_validate, train_neural_net, draw_neural_net

#%%
# --- Data Loading and Initial Preprocessing ---
print("___Data Loading and Initial Preprocessing___")
abalone = fetch_ucirepo(id=1)
X = abalone.data.features
y = abalone.data.targets
key=abalone.data.keys()
print("key", key)

raw_data = X.values         # Convert DataFrame to a raw NumPy array
print(abalone.metadata)     # Print dataset metadata
print(abalone.variables)    # Print variable information


X['Male'] = (X.iloc[:, 0] == 'M').astype(int)
X['Female'] = (X.iloc[:, 0] == 'F').astype(int)
X['Infant'] = (X.iloc[:, 0] == 'I').astype(int)
X = X.drop(columns=[X.columns[0]])

attributeNames = np.asarray(X.columns).tolist()
classLabels = y.values[:, 0]  # Extract class labels from the target DataFrame
classNames = np.unique(classLabels).tolist()  # Get the unique class names
classDict = dict(zip(classNames, range(1, len(classNames) + 1)))  # Map classes to integer values

N, M = X.shape
C = len(classNames)

yy = y.values.squeeze() # squeeze to convert to 1D array


#%%
# --- Normalize Features Globally Before CV ---
print("____Globally Normalizing Features___")
# xx = np.empty((N, M))
# for i in range(M):
#     xx[:, i] = np.array(X.iloc[:, i]).T
# # Standardize data by subtracting the mean and dividing by the standard deviation for each feature
# x_normalized = np.empty((N, M))
# for i in range(M):
#     x_normalized[:, i] = (xx[:, i] - np.mean(xx[:, i])) / np.std(xx[:, i])
pd.set_option('display.max_columns', None)
x_normalized= stats.zscore(X)
print("x_normalized\n",pd.DataFrame(x_normalized, columns=attributeNames))
#%%
# --- Part A: Regularized Linear Regression with Single CV ---
print("_____Part A: Regularized Linear Regression_____")

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

k = 0
for k, (train_index, test_index) in enumerate(CV.split(x_normalized, yy)):
    print('Cross validation fold {0}/{1}:'.format(k + 1, K))
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

    "w* = (X^T X + λI)⁻¹ X^T y" "in lecture page 7 and after"
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
    "w* = (X^T X + λI)⁻¹ X^T y" "in lecture page 7 and after"
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


    # Display the results for the last cross-validation fold
    if k == K - 1:
        plt.figure(k, figsize=(12, 8))
        plt.subplot(1, 2, 1)
        plt.semilogx(lambdas, mean_w_vs_lambda.T[:, 1:], ".-")  # Don't plot the bias term
        plt.xlabel("Regularization factor")
        plt.ylabel("Mean Coefficient Values")
        plt.grid()

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

    k += 1

plt.show()
# Display results
print("Linear regression without feature selection:")
print("- Training error: {0}".format(Error_train.mean()))
print("- Test error:     {0}".format(Error_test.mean()))
print("- R^2 train:     {0}".format((Error_train_nofeatures.sum() - Error_train.sum())
                                    / Error_train_nofeatures.sum()))
print("- R^2 test:     {0}\n".format((Error_test_nofeatures.sum() - Error_test.sum())
                                     / Error_test_nofeatures.sum()))
print("Regularized linear regression:")
print("- Training error: {0}".format(Error_train_rlr.mean()))
print("- Test error:     {0}".format(Error_test_rlr.mean()))
print( "- R^2 train:     {0}".format((Error_train_nofeatures.sum() - Error_train_rlr.sum())
        / Error_train_nofeatures.sum())
)
print("- R^2 test:     {0}\n".format(
        (Error_test_nofeatures.sum() - Error_test_rlr.sum())
        / Error_test_nofeatures.sum()
    )
)

print("Weights in last fold:")
for m in range(M):
    print("{:>15} {:>15}".format(attributeNames[m], np.round(w_rlr[m, -1], 2)))

print("Ran Exercise 8.1.1")
# Get weights and labels (skip the offset/bias term)
last_fold_weights = w_rlr[1:, -1]  # Exclude offset
feature_labels = attributeNames[1:]  # Exclude offset label

# Sort by absolute weight
sorted_indices = np.argsort(np.abs(last_fold_weights))[::-1]
sorted_weights = last_fold_weights[sorted_indices]
sorted_features = [feature_labels[i] for i in sorted_indices]

# Plot
plt.figure(figsize=(8, 6))
plt.barh(sorted_features, sorted_weights)
plt.xlabel("Mean feature weights")
plt.title("Feature weights")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

print("______Regression, part A Done____")


#%% --- Part B:  Regression Two-Level Cross-Validation with ANN and Comparison ---

print("____Part B: Two-Level Cross-Validation___")

"""The actual flow is:
for outer_fold in K1:
    inner_cv = KFold(K2)
    for param in parameters:
        for inner_fold in K2:
            train model on Dtrain
            validate on Dval
        compute average val error
    select best param
    train final model on Dpar
    test on Dtest
"""

" add offset column here "
# N, M = x_normalized.shape
# print("x_normalized\n",pd.DataFrame(x_normalized, columns=attributeNames).assign(Target=yy))

"Exclude Offset column"
x_normalized= x_normalized[:, 1:]  # Exclude Offset column
N, M = x_normalized.shape  # Exclude Offset column
attributeNames = list(X.columns)

# Set parameters
K1, K2 = 10, 5
outer_cv = model_selection.KFold(K1, shuffle=True, random_state=1)
lambdas = np.power(10.0, range(-5, 0))
n_hidden_units_list = [1,2,3,5]
max_iter = 10000
n_replicates = 1

results = []
all_learning_curves = []
all_ann_errors = []

# Begin outer cross-validation
for k, (outer_train_idx, outer_test_idx) in enumerate(outer_cv.split(x_normalized, yy)):
    print(f"\nCrossvalidation fold: {k + 1}/{K1}")
    X_train_outer, y_train_outer = x_normalized[outer_train_idx], yy[outer_train_idx]
    X_test_outer, y_test_outer = x_normalized[outer_test_idx], yy[outer_test_idx]

    # --- Inner CV for RLR ---
    inner_cv = model_selection.KFold(K2, shuffle=True)
    rlr_val_errors = []
    for lmbd in lambdas:
        inner_errors = []
        for inner_train_idx, inner_val_idx in inner_cv.split(X_train_outer):
            X_inner_train = X_train_outer[inner_train_idx]
            y_inner_train = y_train_outer[inner_train_idx]
            X_inner_val = X_train_outer[inner_val_idx]
            y_inner_val = y_train_outer[inner_val_idx]

            M = X_inner_train.shape[1]
            X_train_bias = np.concatenate((np.ones((X_inner_train.shape[0], 1)), X_inner_train), axis=1)
            X_val_bias = np.concatenate((np.ones((X_inner_val.shape[0], 1)), X_inner_val), axis=1)
            lambdaI = lmbd * np.eye(M + 1)
            lambdaI[0, 0] = 0
            w_rlr = np.linalg.solve(X_train_bias.T @ X_train_bias + lambdaI, X_train_bias.T @ y_inner_train)
            val_error = np.mean((y_inner_val - X_val_bias @ w_rlr) ** 2)
            inner_errors.append(val_error)
        rlr_val_errors.append(np.mean(inner_errors))
    # Select optimal lambda based on validation errors
    opt_lambda = lambdas[np.argmin(rlr_val_errors)]
    # Retrain RLR on full outer training set
    M_rlr = X_train_outer.shape[1]
    X_train_rlr = np.concatenate((np.ones((X_train_outer.shape[0], 1)), X_train_outer), axis=1)
    X_test_rlr = np.concatenate((np.ones((X_test_outer.shape[0], 1)), X_test_outer), axis=1)
    lambdaI = opt_lambda * np.eye(M_rlr + 1)
    lambdaI[0, 0] = 0
    w_rlr = np.linalg.solve(X_train_rlr.T @ X_train_rlr + lambdaI, X_train_rlr.T @ y_train_outer)
    # Compute test error for linear regression over the outer test set
    rlr_test_error = np.mean((y_test_outer - X_test_rlr @ w_rlr) ** 2)

    # --- Inner CV for ANN ---
    ann_val_errors = []
    for h in n_hidden_units_list:
        inner_errors = []
        for inner_train_idx, inner_val_idx in inner_cv.split(X_train_outer):
            X_inner_train = torch.Tensor(X_train_outer[inner_train_idx])
            y_inner_train = torch.Tensor(y_train_outer[inner_train_idx]).unsqueeze(1)
            X_inner_val = torch.Tensor(X_train_outer[inner_val_idx])
            y_inner_val = torch.Tensor(y_train_outer[inner_val_idx]).unsqueeze(1)

            model = lambda: torch.nn.Sequential(
                torch.nn.Linear(M_rlr, h), torch.nn.ReLU(), torch.nn.Linear(h, 1)
            )
            net, _, _ = train_neural_net(model, torch.nn.MSELoss(), X=X_inner_train, y=y_inner_train, n_replicates=n_replicates, max_iter=max_iter)
            pred = net(X_inner_val)
            val_loss = torch.nn.MSELoss()(pred, y_inner_val).item()
            inner_errors.append(val_loss)
        ann_val_errors.append(np.mean(inner_errors))

    best_h = n_hidden_units_list[np.argmin(ann_val_errors)]
    model = lambda: torch.nn.Sequential(
        torch.nn.Linear(M_rlr, best_h), torch.nn.ReLU(), torch.nn.Linear(best_h, 1)
    )
    net, _, learning_curve = train_neural_net(model, torch.nn.MSELoss(),
        X=torch.Tensor(X_train_outer).float(),
        y=torch.Tensor(y_train_outer).unsqueeze(1).float(),
        n_replicates=n_replicates, max_iter=max_iter)
    ann_test_error = torch.nn.MSELoss()(net(torch.Tensor(X_test_outer).float()),
                                        torch.Tensor(y_test_outer).unsqueeze(1).float()).item()

    baseline_pred = np.mean(y_train_outer)
    baseline_error = np.mean((y_test_outer - baseline_pred) ** 2)

    results.append({
        'Fold': k + 1,
        'ANN hidden units': best_h,
        'ANN error': np.round(ann_test_error, 3),
        'RLR lambda': np.round(opt_lambda, 3),
        'RLR error': np.round(rlr_test_error, 3),
        'Baseline error': np.round(baseline_error,3)
    })

    all_learning_curves.append(learning_curve)
    all_ann_errors.append(ann_test_error)


# --- Plot ANN Learning Curves ---
plt.figure(figsize=(12, 5))
for i, curve in enumerate(all_learning_curves):
    plt.plot(curve, label=f'Fold {i+1}')
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("ANN Learning Curves (All Outer Folds)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# --- Diagram of Best Neural Net in Last Fold ---
print("\nDiagram of best neural net in last fold:")
weights = [net[i].weight.data.numpy().T for i in [0, 2]]
biases = [net[i].bias.data.numpy() for i in [0, 2]]
tf = [str(net[i]) for i in [1, 2]]
attributeNames_full = attributeNames  # No offset
try:
    from dtuimldmtools import draw_neural_net
    draw_neural_net(weights, biases, tf, attribute_names=attributeNames_full)
except Exception as e:
    print(f"Could not draw neural net diagram: {e}")

# --- Bar Plot of ANN MSEs Across Folds ---
plt.figure(figsize=(8, 5))
plt.bar(np.arange(1, K1 + 1), all_ann_errors, color='skyblue')
plt.xlabel("Fold")
plt.ylabel("MSE")
plt.title("Test Mean Squared Error (ANN Model)")
plt.xticks(np.arange(1, K1 + 1))
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# --- Prediction vs True for Last Fold ---
y_est = net(torch.Tensor(X_test_outer)).detach().numpy().squeeze()
y_true = np.array(y_test_outer).squeeze()
axis_range = [min(y_est.min(), y_true.min()) - 1, max(y_est.max(), y_true.max()) + 1]
plt.figure(figsize=(6, 6))
plt.plot(axis_range, axis_range, 'k--')
plt.plot(y_true, y_est, 'ob', alpha=0.5)
plt.xlabel("True")
plt.ylabel("Estimated")
plt.title("ANN Prediction vs. True (Last Fold)")
plt.grid()
plt.show()

# --- Save and Show Results ---
results_df = pd.DataFrame(results)
results_df.to_csv('results.csv', index=False)
print("\n  Regression, part b: Two-level cross-validation table used to compare the three models:")
print(results_df)
print("\nAverage Test Errors:")

print(results_df[['ANN error', 'RLR error', 'Baseline error']].mean())
plt.figure(figsize=(10, 6))
plt.plot(results_df['Fold'], results_df['ANN error'], label='ANN Test Error', marker='o')
plt.plot(results_df['Fold'], results_df['RLR error'], label='Linear Regression Test Error', marker='s')
plt.plot(results_df['Fold'], results_df['Baseline error'], label='Baseline Error', linestyle='--', color='gray')
plt.xlabel("Fold")
plt.ylabel("Test Error Rate (%)")
plt.title("Classification Error per Fold")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
