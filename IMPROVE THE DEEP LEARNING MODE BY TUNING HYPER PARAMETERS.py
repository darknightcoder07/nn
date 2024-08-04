import numpy as np
from sklearn.datasets import make_classification
from scipy.stats import randint
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_classes=2, random_state=42)
param_dist = {
    "max_depth": [3, None],
    "max_features": randint(1, 20),  
    "min_samples_leaf": randint(1, 10),  
    "criterion": ["gini", "entropy"]
}
tree = DecisionTreeClassifier()
tree_cv = RandomizedSearchCV(tree, param_dist, cv=5, n_iter=100, random_state=42)
tree_cv.fit(X, y)
print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))
print("Best score is {}".format(tree_cv.best_score_))
