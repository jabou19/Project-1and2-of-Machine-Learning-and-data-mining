
# --- Imports and Setup ---
#%%
from ucimlrepo import fetch_ucirepo
import numpy as np
import pandas as pd
import torch
from scipy import stats
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from dtuimldmtools import train_neural_net, draw_neural_net


# Load and process data
abalone = fetch_ucirepo(id=1)
X = abalone.data.features
y = abalone.data.targets.squeeze() # y is Rings in dataset
X['Rings'] = y  # Add Rings as a column in X for future reference
N, M = X.shape
print("X.shape\n", X)


raw_data = X.values         # Convert DataFrame to a raw NumPy array
print(abalone.metadata)     # Print dataset metadata
print(abalone.variables)

# Remove ambiguous class 'I'
X = X[X['Sex'] != 'I']
y = y.loc[X.index]

# Binary classification target
y = X['Sex'].map({'F': 1, 'M': 0}).values.astype(np.float32)

" Drop only 'Sex' from features"
X = X.drop(columns=['Sex'])
N, M = X.shape
attributeNames = X.columns.tolist()  # Exclude 'Sex' from headers

X=X.values

pd.set_option('display.max_columns', None)  # Ensure all columns are displayed


y = y.squeeze() # Ensure y is a 1D array

#%%
print("____Globally Normalizing Features___")
#The line X = stats.zscore(X) is used to normalize the dataset X by applying z-score normalization (also known as standardization).
#This transformation ensures that each feature in X has:
#A mean of 0
#A standard deviation of 1
X= stats.zscore(X)
print("X Normalizing data set:\n", pd.DataFrame(X, columns=attributeNames[:M]))


#%%
# --- Two-Level Cross-Validation (ANN vs Logistic Regression vs Baseline) ---
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
print("\nClassification part")
print("\nClassification: Two-level cross-validation  used to compare the three models in the classification problem.")
print(" ANN vs Logistic Regression vs Baseline")
K1, K2 = 10, 5  # Outer and inner folds
outer_cv = model_selection.KFold(K1, shuffle=True, random_state=1)
lambdas = np.power(10.0, np.arange(-5, 0))

n_hidden_units_list = [1,2,3,5]
max_iter = 10000
n_replicates = 1

results = []
all_learning_curves = []  # For ANN learning curve plot

# Outer cross-validation
for k, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
    print(f"\nOuter Fold {k + 1}/{K1}")
    X_train_outer, y_train_outer = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    #____ Inner CV for ANN-----
    best_h, best_val_error = None, np.inf
    # Inner cross-validation
    inner_cv = model_selection.KFold(K2, shuffle=True)
    # Loop over hidden units
    for h in n_hidden_units_list:
        inner_errors = []
        for inner_train_idx, inner_val_idx in inner_cv.split(X_train_outer):
            X_in_train = torch.tensor(X_train_outer[inner_train_idx]).float()
            y_in_train = torch.tensor(y_train_outer[inner_train_idx]).unsqueeze(1)
            X_val = torch.tensor(X_train_outer[inner_val_idx]).float()
            y_val = torch.tensor(y_train_outer[inner_val_idx]).unsqueeze(1)
            # Define the ANN model in inner
            model = lambda: torch.nn.Sequential(
                torch.nn.Linear(M, h),
                torch.nn.Tanh(),
                torch.nn.Linear(h, 1),
                torch.nn.Sigmoid()
            )

            net, _, _ = train_neural_net(model, torch.nn.BCELoss(), X=X_in_train, y=y_in_train, n_replicates=n_replicates, max_iter=max_iter,tolerance=1e-6)
            y_val_pred = (net(X_val) > 0.5).int()
            val_error = (y_val_pred != y_val.int()).float().mean().item()
            inner_errors.append(val_error)

        avg_error = np.mean(inner_errors)
        if avg_error < best_val_error:
            best_val_error = avg_error
            best_h = h

    # Train best ANN on full outer training set
    final_model = lambda: torch.nn.Sequential(
        torch.nn.Linear(M, best_h),
        torch.nn.Tanh(),
        torch.nn.Linear(best_h, 1),
        torch.nn.Sigmoid()
    )

    net, _, learning_curve = train_neural_net(
        final_model, torch.nn.BCELoss(),
        X=torch.tensor(X_train_outer).float(),
        y=torch.tensor(y_train_outer).unsqueeze(1),
        n_replicates=n_replicates, max_iter=max_iter,tolerance=1e-6
    )
    all_learning_curves.append(learning_curve)
    y_ann_pred = (net(torch.tensor(X_test).float()) > 0.5).int().numpy().squeeze()
    # Test error for ANN for the outer fold
    ann_test_error = np.mean(y_ann_pred != y_test)

    # _____Logistic Regression______
    best_lambda, best_lr_error = None, np.inf
    for lmbd in lambdas:
        lr_errors = []
        for inner_train_idx, inner_val_idx in inner_cv.split(X_train_outer):
            X_in_train, y_in_train = X_train_outer[inner_train_idx], y_train_outer[inner_train_idx]
            X_val, y_val = X_train_outer[inner_val_idx], y_train_outer[inner_val_idx]
            # ___Logistic Regression___ .
            lr = LogisticRegression(penalty='l2', C=1 / lmbd, max_iter=max_iter)
            lr.fit(X_in_train, y_in_train)
            pred = lr.predict(X_val)
            lr_errors.append(np.mean(pred != y_val))

        avg_error = np.mean(lr_errors)
        if avg_error < best_lr_error:
            best_lr_error = avg_error
            best_lambda = lmbd

    lr = LogisticRegression(penalty='l2', C=1 / best_lambda, max_iter=max_iter)
    lr.fit(X_train_outer, y_train_outer)
    y_lr_pred = lr.predict(X_test)
    # Test error for Logistic Regression for the outer fold
    lr_test_error = np.mean(y_lr_pred != y_test)

    # Baseline
    majority = int(np.round(y_train_outer.mean()) > 0.5)
    y_base_pred = np.ones_like(y_test) * majority
    baseline_error = np.mean(y_base_pred != y_test)

    results.append({
        'Fold': k + 1,
        'ANN h*': best_h,
        'ANN E_test': np.round(ann_test_error, 3),
        'LR λ*': np.round(best_lambda, 3),
        'LR E_test': np.round(lr_test_error, 3),
        'Baseline E_test': np.round(baseline_error, 3)
    })

# --- Final Results Table and Summary ---
results_df = pd.DataFrame(results)

# --- Combined Figure: ANN Learning Curves and Error Rates ---
fig, summaries_axes = plt.subplots(1, 2, figsize=(12, 5))
color_list = [
    "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
    "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"
]

# Plot learning curves
for k, curve in enumerate(all_learning_curves):
    (line,) = summaries_axes[0].plot(curve, color=color_list[k % len(color_list)])
    line.set_label(f"Fold {k + 1}")
summaries_axes[0].set_xlabel("Iterations")
summaries_axes[0].set_ylabel("Loss")
summaries_axes[0].set_title("ANN Learning Curves")
summaries_axes[0].legend()
summaries_axes[0].grid()

# Plot ANN test error per fold
ann_errors = [entry['ANN E_test'] for entry in results]
summaries_axes[1].bar(np.arange(1, K1 + 1), ann_errors, color=color_list[:K1])
summaries_axes[1].set_xlabel("Fold")
summaries_axes[1].set_ylabel("Error Rate")
summaries_axes[1].set_title("ANN Test Error Rates")
summaries_axes[1].set_xticks(np.arange(1, K1 + 1))
summaries_axes[1].grid(axis='y')

plt.tight_layout()
plt.show()

# --- Diagram of best neural net in last fold ---
print("\nDiagram of best neural net in last fold:")
weights = [net[i].weight.data.numpy().T for i in [0, 2]]
biases = [net[i].bias.data.numpy() for i in [0, 2]]
tf = [str(net[i]) for i in [1, 3]]
draw_neural_net(weights, biases, tf, attribute_names=attributeNames)

# --- Plot 1: Test Error per Fold ---
plt.figure(figsize=(10, 6))
plt.plot(results_df['Fold'], results_df['ANN E_test'], label='ANN Test Error', marker='o')
plt.plot(results_df['Fold'], results_df['LR E_test'], label='Logistic Regression Test Error', marker='s')
plt.plot(results_df['Fold'], results_df['Baseline E_test'], label='Baseline Error', linestyle='--', color='gray')
plt.xlabel("Fold")
plt.ylabel("Test Error Rate (%)")
plt.title("Classification Error per Fold")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
print(results_df)
print("\nAverage Errors Across Folds:")
print(results_df[['ANN E_test', 'LR E_test', 'Baseline E_test']].mean())
print("\nClassification Two-Level CV Results:")
