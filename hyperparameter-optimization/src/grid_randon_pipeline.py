import  pandas as pd
import numpy as np
from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection
from pathlib import Path
from sklearn import preprocessing
from sklearn import decomposition
from sklearn import pipeline

if __name__ == '__main__':
    train_file_path = Path("/Users/intellify-sarojdongol/workspace/scikit-learn/hyperparameter-optimization/input/train.csv")
    df = pd.read_csv(train_file_path)
    X_train = df.drop("price_range", axis=1).values
    y_train = df.price_range.values

    scl = preprocessing.StandardScaler()
    pca = decomposition.PCA()
    rf = ensemble.RandomForestClassifier(n_jobs=-1)

    classifier = pipeline.Pipeline(
        [
            ("scaling", scl),
            ("pca", pca),
            ("rf", rf)

    ]
    ) 

    param_grid = {
        "pca__n_components": np.arange(5,10),
        "rf__n_estimators": np.arange(100, 1500, 100),
        "rf__max_depth": np.arange(1, 3),
        "rf__criterion": ["gini", "entropy"]
    }

    model = model_selection.RandomizedSearchCV(
        estimator=classifier,
        param_distributions=param_grid,
        n_iter=10,
        scoring="accuracy",
        verbose=10,
        n_jobs=1,
        cv=5
    )

    model.fit(X_train, y_train)
    print(model.best_score_)
    print(model.best_estimator_.get_params())