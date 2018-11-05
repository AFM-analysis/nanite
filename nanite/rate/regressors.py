"""scikit-learn regressors and their keyword arguments"""
from sklearn import ensemble, svm, tree


reg_dict = {
    "AdaBoost": [
        ensemble.AdaBoostRegressor,
        {"learning_rate": .5,
         "n_estimators": 100,
         "random_state": 42,
         },
    ],
    "Decision Tree": [
        tree.DecisionTreeRegressor,
        {"max_depth": 6,
         "min_samples_leaf": 4,
         "min_samples_split": 4,
         "random_state": 42,
         }
    ],
    "Extra Trees": [
        ensemble.ExtraTreesRegressor,
        {"max_depth": 15,
         "min_samples_leaf": 2,
         "min_samples_split": 5,
         "n_estimators": 100,
         "random_state": 42,
         }
    ],
    "Gradient Tree Boosting": [
        ensemble.GradientBoostingRegressor,
        {"max_depth": 5,
         "min_samples_leaf": 4,
         "min_samples_split": 7,
         "n_estimators": 100,
         "random_state": 42,
         }
    ],
    "Random Forest": [
        ensemble.RandomForestRegressor,
        {"max_depth": 15,
         "min_samples_leaf": 2,
         "min_samples_split": 7,
         "n_estimators": 100,
         "random_state": 42,
         }
    ],
    "SVR (linear kernel)": [
        svm.LinearSVR,
        {"C": 2,
         "epsilon": 1.0,
         "random_state": 42,
         },
    ],
    "SVR (RBF kernel)": [
        svm.SVR,
        {"kernel": "rbf",
         "C": 25,
         "epsilon": 0.7,
         },
    ],
}


#: List of available default regressor names
reg_names = sorted(reg_dict.keys())

#: List of tree-based regressor class names (used for keyword defaults in
#: :class:`IndentationRater`)
reg_trees = ["AdaBoostRegressor",
             "DecisionTreeRegressor",
             "ExtraTreesRegressor",
             "GradientBoostingRegressor",
             "RandomForestRegressor"]
