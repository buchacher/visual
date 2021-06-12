import pickle

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

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

classifiers = {
    'baseline_lr': LogisticRegression(),
    'dt': DecisionTreeClassifier(),
    'knn': KNeighborsClassifier(),
    'mlp': MLPClassifier(),
    'rf': RandomForestClassifier(),
    'sgd': SGDClassifier(),
    'svc': SVC()
}

results = []
names = []
for classifier in classifiers:
    # Build the model
    model = classifiers[classifier]

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

    # Output current model, target and optimal params for orientation
    print("Current classifier: " + classifier)
    print("Current target: " + target)
    print("Optimal parameters: " + str(optimal_params))

    # Evaluate model performance on the training set using 3-fold cross validation
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    cv_result = cross_val_score(clf, X_train, y_train, scoring='balanced_accuracy', cv=skf)
    names.append(classifier)
    results.append(cv_result)
    print("Training set CV mean accuracy: %.3f" % cv_result.mean())

# Boxplot
fig = plt.figure()
fig.suptitle("CV Model Evaluation on Training Set for " + target)
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


"""
Current classifier: baseline_lr
Current target: color
Optimal parameters: {'classifier__C': 0.001, 'classifier__class_weight': 'balanced', 'classifier__max_iter': 1000, 'classifier__multi_class': 'multinomial', 'classifier__random_state': 0, 'classifier__solver': 'newton-cg', 'ftselector__k': 50, 'prepper__drop_id_bounding': False, 'prepper__filter_characteristics': 'none'}
Training set CV mean accuracy: 0.215
Current classifier: dt
Current target: color
Optimal parameters: {'classifier__ccp_alpha': 0.1, 'classifier__class_weight': 'balanced', 'classifier__criterion': 'entropy', 'classifier__random_state': 0, 'ftselector__k': 100, 'prepper__drop_id_bounding': False, 'prepper__filter_characteristics': 'none'}
Training set CV mean accuracy: 0.200
Current classifier: knn
Current target: color
Optimal parameters: {'classifier__n_jobs': -1, 'classifier__n_neighbors': 5, 'classifier__p': 1, 'ftselector__k': 20, 'prepper__drop_id_bounding': False, 'prepper__filter_characteristics': 'none'}
Training set CV mean accuracy: 0.161
Current classifier: mlp
Current target: color
Optimal parameters: {'classifier__activation': 'tanh', 'classifier__batch_size': 100, 'classifier__early_stopping': True, 'classifier__hidden_layer_sizes': (200, 200), 'classifier__learning_rate_init': 0.001, 'classifier__momentum': 0.9, 'classifier__n_iter_no_change': 10, 'classifier__random_state': 0, 'classifier__solver': 'sgd', 'ftselector__k': 200, 'prepper__drop_id_bounding': False, 'prepper__filter_characteristics': 'hog'}
Training set CV mean accuracy: 0.187
Current classifier: rf
Current target: color
Optimal parameters: {'classifier__class_weight': None, 'classifier__criterion': 'gini', 'classifier__n_estimators': 100, 'classifier__n_jobs': -1, 'classifier__oob_score': True, 'classifier__random_state': 0, 'ftselector__k': 100, 'prepper__drop_id_bounding': False, 'prepper__filter_characteristics': 'none'}
Training set CV mean accuracy: 0.181
Current classifier: sgd
Current target: color
Optimal parameters: {'classifier__alpha': 0.01, 'classifier__early_stopping': True, 'classifier__loss': 'modified_huber', 'classifier__n_iter_no_change': 10, 'classifier__penalty': 'l1', 'classifier__random_state': 0, 'ftselector__k': 200, 'prepper__drop_id_bounding': False, 'prepper__filter_characteristics': 'none'}
Training set CV mean accuracy: 0.187
Current classifier: svc
Current target: color
Optimal parameters: {'classifier__C': 1.0, 'classifier__class_weight': 'balanced', 'classifier__gamma': 'scale', 'classifier__kernel': 'linear', 'classifier__random_state': 0, 'ftselector__k': 10, 'prepper__drop_id_bounding': False, 'prepper__filter_characteristics': 'hog'}
Training set CV mean accuracy: 0.209
"""