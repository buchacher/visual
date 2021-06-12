import pickle

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from data_prep import StacsTransformer

'''~Config'''
target = 'color'

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

# Assign classifier name
classifier = 'rf'

# Build the model
model = RandomForestClassifier()

# Combine preprocessor and model into one pipeline
clf = Pipeline(steps=[
    ('prepper', StacsTransformer()),
    ('preprocessor', preprocessor),
    ('ftselector', SelectKBest(f_classif)),
    ('classifier', model)
])

# Load optimal pipeline parameters
path = 'optimal-params/' + classifier + '_' + target + '.pkl'
with open(path, 'rb') as f:
    optimal_params = pickle.load(f)

# Set optimal pipeline parameters
clf.set_params(**optimal_params)

# Fit classifier on training set
clf_fit = clf.fit(X_train, y_train)

# Load test using pandas
X_test = pd.read_csv('data/data_test.csv')

# Drop one row with missing values
X_test.dropna(inplace=True)

# Set index to 'id' column as insignificant as a feature
X_test = X_test.set_index('id')

y_hat = clf_fit.predict(X_test)

if target == 'color':
    path = 'output/colour_test.csv'
else:
    path = 'output/texture_test.csv'

# Output corresponding predictions to a text file
np.savetxt(path, y_hat, fmt='%s')
