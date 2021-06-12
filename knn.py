import os
import pickle

import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from data_prep import StacsTransformer

'''~Config'''
target = 'texture'

# Load data using pandas
df = pd.read_csv('data/data_train.csv')

# Drop target we are not predicting
if target == 'color':
    df = df.drop(columns=['texture'])
else:
    df = df.drop(columns=['color'])

# Drop one row with missing values
df.dropna(inplace=True)

# Pop out respective target class
y = df.pop(target)

# Assign input (excludes output classes)
X = df

# Split into training and validation set, stratify with regard to target
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

# Set index to 'id' column as insignificant as a feature
X_train = X_train.set_index('id')
X_valid = X_valid.set_index('id')

# Build transformer for categorical features
cat_transformer = Pipeline([
    ('onehot', OneHotEncoder())
])

# Build transformer for numeric features
num_transformer = Pipeline([
    ('scaler', StandardScaler())
])

# Combine transformers
preprocessor = ColumnTransformer(transformers=[
    ('cat', cat_transformer, make_column_selector(dtype_include='object')),
    ('num', num_transformer, make_column_selector(dtype_include=['float64', 'int64']))
])

# Build the model
model = KNeighborsClassifier()

# Combine preprocessor and model into one pipeline
clf = Pipeline(steps=[
    ('prepper', StacsTransformer()),
    ('preprocessor', preprocessor),
    ('ftselector', SelectKBest(f_classif)),
    ('classifier', model)
])

params = {
    'prepper__filter_characteristics': ['none', 'hog', 'bimp', 'cielab'],
    'prepper__drop_id_bounding': [False, True],
    'ftselector__k': [10, 20, 27, 32, 50, 100, 200, 400, 440, 'all'],
    'classifier__n_neighbors': [3, 5, 7, 11],
    'classifier__p': [1, 2],
    'classifier__n_jobs': [-1]
}

# Instantiate folds and grid search
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
grid_search = GridSearchCV(clf, params, scoring='balanced_accuracy', cv=skf, verbose=4)

# Fit classifier on training set
clf_fit = grid_search.fit(X_train, y_train)

# Output current target for orientation
print("Current target: " + target)

# Get predicted labels and output balanced accuracy on training set
y_hat_train = clf_fit.predict(X_train)
train_bal_accuracy = balanced_accuracy_score(y_train, y_hat_train)
print("Training set balanced accuracy: %.3f" % train_bal_accuracy)

# Get and output optimal params
optimal_params = clf_fit.best_params_
print("Grid search - optimal params:")
print(clf_fit.best_params_)

# Save optimal pipeline args to disk
path = 'optimal-params/' + os.path.basename(__file__).split('.')[0] + '_' + target + '.pkl'
with open(path, 'wb') as f:
    pickle.dump(optimal_params, f)

"""
Current target: color
Training set balanced accuracy: 0.232
Grid search - optimal params:
{'classifier__n_jobs': -1, 'classifier__n_neighbors': 5, 'classifier__p': 1, 'ftselector__k': 20, 'prepper__drop_id_bounding': False, 'prepper__filter_characteristics': 'none'}

Current target: texture
Training set balanced accuracy: 0.464
Grid search - optimal params:
{'classifier__n_jobs': -1, 'classifier__n_neighbors': 3, 'classifier__p': 1, 'ftselector__k': 200, 'prepper__drop_id_bounding': False, 'prepper__filter_characteristics': 'hog'}
"""