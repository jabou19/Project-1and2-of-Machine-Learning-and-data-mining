#%%
# --- Imports and Setup ---
from ucimlrepo import fetch_ucirepo
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn import model_selection
from dtuimldmtools import rlr_validate, train_neural_net, draw_neural_net

#%%
# --- Data Loading and Initial Preprocessing ---
abalone = fetch_ucirepo(id=1)
X = abalone.data.features
y = abalone.data.targets

X['Male'] = (X.iloc[:, 0] == 'M').astype(int)
X['Female'] = (X.iloc[:, 0] == 'F').astype(int)
X['Infant'] = (X.iloc[:, 0] == 'I').astype(int)
X = X.drop(columns=[X.columns[0]])

attributeNames = np.asarray(X.columns).tolist()
N, M = X.shape
yy = y.values.squeeze()
xx = X.values

#%%
# --- Normalize Features Globally Before CV ---
xx = (xx - np.mean(xx, axis=0)) / np.std(xx, axis=0)

#%%
# --- Part A: Regularized Linear Regression with Single CV ---
print("Part A: Regularized Linear Regression")

xx_offset = np.concatenate((np.ones((N, 1)), xx), axis=1)
attributeNames_offset = ["Offset"] + attributeNames
M_offset = M + 1

K = 10
CV = model_selection.KFold(K, shuffle=True, random_state=1)
lambdas = np.power(10.0, range(-5, 9))

Error_train = np.empty(K)
Error_test = np.empty(K)
Error_train_rlr = np.empty(K)
Error_test_rlr = np.empty(K)
Error_train_nofeatures = np.empty(K)
Error_test_nofeatures = np.empty(K)

mu = np.empty((K, M))
sigma = np.empty((K, M))
w_rlr = np.empty((M_offset, K))
w_noreg = np.empty((M_offset, K))

for k, (train_idx, test_idx) in enumerate(CV.split(xx_offset, yy)):
    print(f"Crossvalidation fold: {k + 1}/{K}")
    X_train = xx_offset[train_idx]
    y_train = yy[train_idx]
    X_test = xx_offset[test_idx]
    y_test = yy[test_idx]

    mu[k, :] = np.mean(X_train[:, 1:], axis=0)
    sigma[k, :] = np.std(X_train[:, 1:], axis=0)

    X_train[:, 1:] = (X_train[:, 1:] - mu[k, :]) / sigma[k, :]
    X_test[:, 1:] = (X_test[:, 1:] - mu[k, :]) / sigma[k, :]

    internal_cv = 10
    opt_val_err, opt_lambda, mean_w, train_err, val_err = rlr_validate(X_train, y_train, lambdas, internal_cv)

    lambdaI = opt_lambda * np.eye(M_offset)
    lambdaI[0, 0] = 0
    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train

    w_rlr[:, k] = np.linalg.solve(XtX + lambdaI, Xty)
    w_noreg[:, k] = np.linalg.solve(XtX, Xty)

    Error_train_nofeatures[k] = np.square(y_train - y_train.mean()).sum() / len(y_train)
    Error_test_nofeatures[k] = np.square(y_test - y_test.mean()).sum() / len(y_test)

    Error_train_rlr[k] = np.square(y_train - X_train @ w_rlr[:, k]).sum() / len(y_train)
    Error_test_rlr[k] = np.square(y_test - X_test @ w_rlr[:, k]).sum() / len(y_test)

    Error_train[k] = np.square(y_train - X_train @ w_noreg[:, k]).sum() / len(y_train)
    Error_test[k] = np.square(y_test - X_test @ w_noreg[:, k]).sum() / len(y_test)

    if k == K - 1:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.semilogx(lambdas, mean_w.T[:, 1:], '.-')
        plt.xlabel('Regularization factor lambda')
        plt.ylabel('Mean Coefficient Values')
        plt.title('Mean Coefficient Paths (No Offset)')
        plt.grid()

        plt.subplot(1, 2, 2)
        plt.loglog(lambdas, train_err.T, 'b.-', lambdas, val_err.T, 'r.-')
        plt.xlabel('Regularization factor lambda')
        plt.ylabel('Squared error')
        plt.title(f'Train vs Validation Error (Lambda*) = 1e{np.log10(opt_lambda):.0f}')
        plt.legend(['Train error', 'Validation error'])
        plt.grid()
        plt.tight_layout()
        plt.show()

print("Linear regression without feature selection:")
print("- Training error:", Error_train.mean())
print("- Test error:", Error_test.mean())
print("- R^2 train:", (Error_train_nofeatures.sum() - Error_train.sum()) / Error_train_nofeatures.sum())
print("- R^2 test:", (Error_test_nofeatures.sum() - Error_test.sum()) / Error_test_nofeatures.sum())

print("Regularized linear regression:")
print("- Training error:", Error_train_rlr.mean())
print("- Test error:", Error_test_rlr.mean())
print("- R^2 train:", (Error_train_nofeatures.sum() - Error_train_rlr.sum()) / Error_train_nofeatures.sum())
print("- R^2 test:", (Error_test_nofeatures.sum() - Error_test_rlr.sum()) / Error_test_nofeatures.sum())

print("Weights in last fold:")
for m in range(M):
    print(f"{attributeNames[m]:>15} {np.round(w_rlr[m, -1], 2):>15}")

print("Regression, part A Done")

#%%
# --- Part B: Two-Level Cross-Validation with ANN and Comparison ---
print("\nPart B: Two-Level Cross-Validation")

x_normalized = xx
x_normalized = np.concatenate((np.ones((N, 1)), x_normalized), axis=1)
M = x_normalized.shape[1]

n_hidden_units_list = [1, 2, 3, 4, 5, 6]
max_iter = 5000
n_replicates = 1
K1, K2 = 5, 5
outer_cv = model_selection.KFold(K1, shuffle=True, random_state=1)
lambdas = np.power(10.0, np.arange(-5, 9))

results = []
all_learning_curves = []
all_ann_errors = []

for k, (outer_train_idx, outer_test_idx) in enumerate(outer_cv.split(x_normalized)):
    print(f"\nCrossvalidation fold: {k + 1}/{K1}")
    X_train_outer, y_train_outer = x_normalized[outer_train_idx], yy[outer_train_idx]
    X_test_outer, y_test_outer = x_normalized[outer_test_idx], yy[outer_test_idx]

    best_h, best_ann_val = None, float('inf')
    for h in n_hidden_units_list:
        inner_errors = []
        inner_cv = model_selection.KFold(K2, shuffle=True)
        for inner_train_idx, inner_val_idx in inner_cv.split(X_train_outer):
            X_inner_train = torch.Tensor(X_train_outer[inner_train_idx])
            y_inner_train = torch.Tensor(y_train_outer[inner_train_idx]).unsqueeze(1)
            X_inner_val = torch.Tensor(X_train_outer[inner_val_idx])
            y_inner_val = torch.Tensor(y_train_outer[inner_val_idx]).unsqueeze(1)

            model = lambda: torch.nn.Sequential(
                torch.nn.Linear(M, h), torch.nn.ReLU(), torch.nn.Linear(h, 1)
            )
            net, _, _ = train_neural_net(model, torch.nn.MSELoss(), X_inner_train, y_inner_train, n_replicates, max_iter)
            pred = net(X_inner_val)
            val_loss = torch.nn.MSELoss()(pred, y_inner_val).item()
            inner_errors.append(val_loss)

        if np.mean(inner_errors) < best_ann_val:
            best_ann_val = np.mean(inner_errors)
            best_h = h

    model = lambda: torch.nn.Sequential(
        torch.nn.Linear(M, best_h), torch.nn.ReLU(), torch.nn.Linear(best_h, 1)
    )
    net, _, learning_curve = train_neural_net(model, torch.nn.MSELoss(),
        X=torch.Tensor(X_train_outer).float(),
        y=torch.Tensor(y_train_outer).unsqueeze(1).float(),
        n_replicates=n_replicates, max_iter=max_iter)

    ann_test_error = torch.nn.MSELoss()(net(torch.Tensor(X_test_outer).float()),
                                        torch.Tensor(y_test_outer).unsqueeze(1).float()).item()

    all_learning_curves.append(learning_curve)
    all_ann_errors.append(ann_test_error)

    opt_val_err, opt_lambda, _, _, _ = rlr_validate(X_train_outer, y_train_outer, lambdas, K2)
    lambdaI = opt_lambda * np.eye(M)
    lambdaI[0, 0] = 0
    w_rlr = np.linalg.solve(X_train_outer.T @ X_train_outer + lambdaI, X_train_outer.T @ y_train_outer)
    rlr_test_error = np.mean((y_test_outer - X_test_outer @ w_rlr) ** 2)

    baseline_pred = np.mean(y_train_outer)
    baseline_error = np.mean((y_test_outer - baseline_pred) ** 2)

    results.append({
        'Fold': k + 1,
        'ANN hidden units': best_h,
        'ANN error': ann_test_error,
        'RLR lambda': opt_lambda,
        'RLR error': rlr_test_error,
        'Baseline error': baseline_error
    })

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
attributeNames_full = ["Offset"] + attributeNames

draw_neural_net(weights, biases, tf, attribute_names=attributeNames_full)

# --- Bar Plot of ANN MSEs Across Folds ---
plt.figure(figsize=(8, 5))
plt.bar(np.arange(1, K1+1), all_ann_errors, color='skyblue')
plt.xlabel("Fold")
plt.ylabel("MSE")
plt.title("Test Mean Squared Error (ANN Model)")
plt.xticks(np.arange(1, K1+1))
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

results_df = pd.DataFrame(results)
results_df.to_csv('results.csv', index=False)
print("\nTwo-level CV Results:")
print(results_df)
print("\nAverage Test Errors:")
print(results_df[['ANN error', 'RLR error', 'Baseline error']].mean())