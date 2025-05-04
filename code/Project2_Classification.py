
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
from dtuimldmtools import mcnemar
# Load and process data
abalone = fetch_ucirepo(id=1)
X = abalone.data.features
y = abalone.data.targets.squeeze() # y is Rings in dataset
X['Rings'] = y  # Add Rings as a column in X for future reference
N, M = X.shape
print("X.shape\n", X)

#key=abalone.data.keys()
#print("key",key)

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
#print("attributeNames", attributeNames)
X=X.values

pd.set_option('display.max_columns', None)  # Ensure all columns are displayed
#print("X data set after dropping the column Sex:\n", pd.DataFrame(X, columns=attributeNames[:M]))

y = y.squeeze() # Ensure y is a 1D array
#print("Unique values in yy:", np.unique(y))
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
For each outer fold:
    Split data into outer training set (D^par_i) and test set (D^test_i)
    For each inner fold:
        Split D^par_i into inner training and validation sets
        
        For each model parameter (e.g., lambda for RLR or h for ANN):
            Train model on inner training set
            Evaluate on inner validation set

    Select the best parameter (lambda* or h*) based on average inner validation error
    Retrain the model on full outer training set (D^par_i) using the best parameter
    Evaluate on outer test set (D^test_i)
"""
print("\nClassification part")
print("\nClassification: Two-level cross-validation  used to compare the three models in the classification problem.")
print(" ANN vs Logistic Regression vs Baseline")
K1, K2 = 10, 5  # Outer and inner folds
outer_cv = model_selection.KFold(K1, shuffle=True, random_state=1)
lambdas = np.power(10.0, np.arange(-5, 1))

n_hidden_units_list = [1,2,3,4,5]
max_iter = 10000
n_replicates = 1

results = []
all_learning_curves = []  # For ANN learning curve plot
y_true_all = []
y_ann_pred_all = []
y_lr_pred_all = []
y_base_pred_all = []

# Outer cross-validation
# loop over outer folds
for k, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
    print(f"\nOuter Fold {k + 1}/{K1}")
    X_train_outer, y_train_outer = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # --- Inner CV for ANN ---
    validation_errors_per_h = {h: [] for h in n_hidden_units_list}
    inner_cv = model_selection.KFold(K2, shuffle=True)
    # Loop over inner folds
    for inner_train_idx, inner_val_idx in inner_cv.split(X_train_outer):
        X_train_in = torch.tensor(X_train_outer[inner_train_idx]).float()
        y_train_in = torch.tensor(y_train_outer[inner_train_idx]).unsqueeze(1)
        X_val = torch.tensor(X_train_outer[inner_val_idx]).float()
        y_val = torch.tensor(y_train_outer[inner_val_idx]).unsqueeze(1)
        # Loop over hidden units
        for h in n_hidden_units_list:
            model = lambda: torch.nn.Sequential(
                torch.nn.Linear(M, h),
                torch.nn.Tanh(),
                torch.nn.Linear(h, 1),
                torch.nn.Sigmoid()
            )
            # Train inner ANN
            net, _, _ = train_neural_net(model, torch.nn.BCELoss(), X_train_in, y_train_in, n_replicates, max_iter=max_iter, tolerance=1e-6)
            y_pred_val = (net(X_val) > 0.5).int()
            err = (y_pred_val != y_val.int()).float().mean().item()
            validation_errors_per_h[h].append(err)
    # Average validation errors for each h
    avg_val_errors = {h: np.mean(errors) for h, errors in validation_errors_per_h.items()}
    # Select best h
    best_h = min(avg_val_errors, key=avg_val_errors.get)

    # Train final ANN
    final_model = lambda: torch.nn.Sequential(
        torch.nn.Linear(M, best_h),
        torch.nn.Tanh(),
        torch.nn.Linear(best_h, 1),
        torch.nn.Sigmoid()
    )
    # Train final ANN on full outer training set
    net, _, learning_curve = train_neural_net(final_model, torch.nn.BCELoss(),
                                              torch.tensor(X_train_outer).float(),
                                              torch.tensor(y_train_outer).unsqueeze(1),
                                              n_replicates, max_iter=max_iter,tolerance= 1e-6)
    y_ann_pred = (net(torch.tensor(X_test).float()) > 0.5).int().numpy().squeeze()
    ann_test_error = np.mean(y_ann_pred != y_test)
    all_learning_curves.append(learning_curve)

    # --- Inner CV for Logistic Regression ---
    validation_errors_per_lambda = {lmbd: [] for lmbd in lambdas}
    # loop over inner folds
    for inner_train_idx, inner_val_idx in inner_cv.split(X_train_outer):
        X_train_in = X_train_outer[inner_train_idx]
        y_train_in = y_train_outer[inner_train_idx]
        X_val = X_train_outer[inner_val_idx]
        y_val = y_train_outer[inner_val_idx]
        # Loop over lambdas
        for lmbd in lambdas:
            lr = LogisticRegression(penalty='l2', C=1 / lmbd, max_iter=100)
            lr.fit(X_train_in, y_train_in)
            pred = lr.predict(X_val)
            err = np.mean(pred != y_val)
            validation_errors_per_lambda[lmbd].append(err)

    avg_val_errors = {lmbd: np.mean(errors) for lmbd, errors in validation_errors_per_lambda.items()}
   # Select best lambda
    best_lambda = min(avg_val_errors, key=avg_val_errors.get)

    # Train on full outer training set
    lr = LogisticRegression(penalty='l2', C=1 / best_lambda, max_iter=100)
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
        'LR λ*': np.round(best_lambda, 8),
        'LR E_test': np.round(lr_test_error, 3),
        'Baseline E_test': np.round(baseline_error, 3)
    })
    # Inside your outer loop, after each prediction block, append like this:
    y_true_all.extend(y_test)
    y_ann_pred_all.extend(y_ann_pred)
    y_lr_pred_all.extend(y_lr_pred)
    y_base_pred_all.extend(y_base_pred)

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
plt.plot(results_df['Fold'], results_df['Baseline E_test'], label='Baseline Error', linestyle='--', color='green')
plt.xlabel("Fold")
plt.ylabel("Test Error Rate")
plt.title("Classification Error per Fold")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
results_df.to_csv('resultsClassification.csv', index=False)
print(results_df)
print("\nAverage Errors Across Folds:")
print(results_df[['ANN E_test', 'LR E_test', 'Baseline E_test']].mean())
print("\nClassification Two-Level CV Results:")

# After the loop, convert to numpy arrays
y_true_all = np.array(y_true_all)
y_ann_pred_all = np.array(y_ann_pred_all)
y_lr_pred_all = np.array(y_lr_pred_all)
y_base_pred_all = np.array(y_base_pred_all)
alpha=0.05
# --- McNemar Tests ---
print("\n--- McNemar Statistical Comparisons ---")

# ANN vs Logistic Regression
print("_____ANN vs Logistic Regression_____")
thetahat, CI, p = mcnemar(y_true_all, y_ann_pred_all, y_lr_pred_all,alpha=alpha)
print("θ̂ = θ_ANN - θ_LR  Point Estimate:", thetahat, "CI:", CI, "p-value:", p)

# ANN vs Baseline
print("______ANN vs Baseline______________")
thetahat, CI, p = mcnemar(y_true_all, y_ann_pred_all, y_base_pred_all,alpha=alpha)
print("θ̂ = θ_ANN - θ_Baseline  Point Estimate:", thetahat, "CI:", CI, "p-value:", p)

# Logistic Regression vs Baseline
print("______Logistic Regression vs Baseline________")
thetahat, CI, p = mcnemar(y_true_all, y_lr_pred_all, y_base_pred_all,alpha=alpha)
print("θ̂ = θ_LR - θ_Baseline  Point Estimate:", thetahat, "CI:", CI, "p-value:", p)