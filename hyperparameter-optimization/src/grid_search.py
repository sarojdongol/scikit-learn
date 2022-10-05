import  pandas as pd
import numpy as np
from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection
from pathlib import Path

"""
Two Types of parameters:

1. Model parameters:
these are the parameters of the model that can be determined by training with training data.
These can be considered as internal parameters.

Examples: Weight and bias
Y =  w * x + b

2. Hyperparameters:

Hyperparameters are parameters whose values control the learning process. These are adjustable parameters
used to obtain an optimal model. External parameters.

A. Learning rate
B. Number of epochs
C. n_estimators : number of decision treen in random forest

Hyperparameter tuning:
refers to the process of choosing the optimun set of hyperparameters for a machine learinging model.
This process is also called Hyperparameter optimization.
Hyperparameter tunning --> Best parameters

Important Technique:

GridSearchCV: uses all the hyperparameters value
RandomSearchCV: randomly selects the hyperparameters value.


Model training --> Best model parameters

"""
if __name__ == '__main__':
    train_file_path = Path("/Users/intellify-sarojdongol/workspace/scikit-learn/hyperparameter-optimization/input/train.csv")
    df = pd.read_csv(train_file_path)
    X_train = df.drop("price_range", axis=1).values
    y_train = df.price_range.values
    
    classifier = ensemble.RandomForestClassifier(n_jobs=-1)
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [1, 3],
        "criterion": ["gini", "entropy"]
    }

    model = model_selection.GridSearchCV(
        estimator=classifier,
        param_grid=param_grid,
        scoring="accuracy",
        verbose=10,
        n_jobs=1,
        cv=5
    )

    model.fit(X_train, y_train)
    print(model.best_score_)
    print(model.best_estimator_.get_params())