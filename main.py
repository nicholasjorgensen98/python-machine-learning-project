# compares two similar sets of data and uses multiple, models, graphs, and methods to be able to tell the two sets are similar
# you need to install numpy pandas matplotlib and scipy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# gets the moderls from sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, RocCurveDisplay

from scipy.stats import ttest_ind, ks_2samp   # Welch’s t and Kolmogorov–Smirnov

# where I generate the sets of data 

rng = np.random.default_rng(seed=1000)

n_a, n_b = 1000, 1000

mean_alpha, cov_alpha = [2, 2], [[1.0, 0.4], [0.4, 1.0]]
mean_beta, cov_beta = [2, 2], [[1.0, 0.4], [0.4, 1.0]]

data_alpha = rng.multivariate_normal(mean_alpha, cov_alpha, size=n_a)
data_beta = rng.multivariate_normal(mean_beta, cov_beta, size=n_b)

df_setalpha = pd.DataFrame(data_alpha, columns=['x1', 'x2'])
df_setbeta = pd.DataFrame(data_beta, columns=['x1', 'x2'])
df_setalpha['label'] = 0        
df_setbeta['label'] = 1        

df = pd.concat([df_setalpha, df_setbeta], ignore_index=True)
print('Combined shape:', df.shape, '\n')
print(df.groupby('label').describe().T, '\n')

# visual inspection with a scatter plot

plt.figure()
for lbl, marker, name in [(0, 'o', 'Dataset Alpha'), (1, 's', 'Dataset Beta')]:
    subset = df[df.label == lbl]
    plt.scatter(subset.x1, subset.x2, marker=marker, alpha=0.4, label=name)


plt.title('Feature space: Alpha vs Beta')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.tight_layout()
plt.show()

# welch t test to compare the means

print('Mean comparison between Dataset Alpha and Dataset Beta')
feature_cols = df.columns.drop('label')
means_similar = True

for col in feature_cols:
    t_stat, p_val = ttest_ind(df_setalpha[col], df_setbeta[col], equal_var=False)
    diff = df_setalpha[col].mean() - df_setbeta[col].mean()
    print(f'{col:>6}: Δmean = {diff:+.3f},  p = {p_val:.4g}')
    if p_val < 0.05:
        means_similar = False

mean_dist = np.linalg.norm(df_setalpha[feature_cols].mean() - df_setbeta[feature_cols].mean())
print(f'\nEuclidean distance between mean vectors: {mean_dist:.3f}')

if means_similar:
    print('\nConclusion (means): No significant difference at α = 0.05.\n')
else:
    print('\nConclusion (means): At least one feature shows a significant difference (α = 0.05).\n')

# this is where i compare teh distributions with a histogram

print('Distribution comparison (Kolm–Smir)')
for col in feature_cols:
    ks_stat, ks_p = ks_2samp(df_setalpha[col], df_setbeta[col])
    print(f'{col:>6}: KS statistic = {ks_stat:.3f},  p = {ks_p:.4g}')

    # Histogram for visual comparison
    plt.figure()
    plt.hist(df_setalpha[col], bins=30, alpha=0.5, label='Alpha', density=True)
    plt.hist(df_setbeta[col], bins=30, alpha=0.5, label='Beta', density=True)
    plt.title(f'Distribution of {col}: Alpha vs Beta')
    plt.xlabel(col)
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.show()

# this is where I have python train and split the data

X = df[feature_cols].values
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=0
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# this is the section where I model and evaluate

clf = LogisticRegression()
clf.fit(X_train_scaled, y_train)

y_pred = clf.predict(X_test_scaled)
print(f'\nTest accuracy: {accuracy_score(y_test, y_pred):.3f}\n')
print('Confusion matrix (rows = true, cols = pred):\n',
      confusion_matrix(y_test, y_pred), '\n')
print('Classification report:\n', classification_report(y_test, y_pred))



RocCurveDisplay.from_estimator(clf, X_test_scaled, y_test)
plt.title('ROC curve (Alpha vs Beta)')
plt.tight_layout()
plt.show()