import  pandas as pd
import numpy as np
from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection
from pathlib import Path
if __name__ == '__main__':
    train_file_path = Path("/Users/intellify-sarojdongol/workspace/scikit-learn/hyperparameter-optimization/input/train.csv")
    df = pd.read_csv(train_file_path)
    X_train = df.drop("price_range", axis=1).values
    y_train = df.price_range.values
    
    classifier = ensemble.RandomForestClassifier(n_jobs=-1)
    param_grid = {
        "n_estimators": [100, 200, 300, 400],
        "max_depth": [1, 3, 5, 7],
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