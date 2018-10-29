"""scikit-learn regressors and their keyword arguments"""
import numpy as np
from sklearn import ensemble, svm, tree


class AverageTreeRegressor(object):
    def __init__(self, ensemble_names, random_states):
        for n, rs in zip(ensemble_names, random_states):
            kwargs = reg_dict[n][1].copy()
            kwargs["random_state"] = rs
            reg_dict[n][0](**kwargs)

        self.ensemble_regs = [reg_dict[n][0](**reg_dict[n][1])
                              for n in ensemble_names]

    def fit(self, *args, **kwargs):
        [er.fit(*args, **kwargs) for er in self.ensemble_regs]

    def predict(self, *args, **kwargs):
        vals = [er.predict(*args, **kwargs) for er in self.ensemble_regs]
        return np.mean(np.array(vals), axis=0)


reg_dict = {
    "AdaBoost": [
        ensemble.AdaBoostRegressor,
        {"learning_rate": .5,
         "n_estimators": 30},
    ],
    "Decision Tree": [
        tree.DecisionTreeRegressor,
        {"max_depth": 6,
         "min_samples_leaf": 4,
         "min_samples_split": 4,
         "random_state": 42}
    ],
    "Extra Trees": [
        ensemble.ExtraTreesRegressor,
        {"max_depth": 15,
         "min_samples_leaf": 2,
         "min_samples_split": 5,
         "random_state": 42}
    ],
    "Gradient Tree Boosting": [
        ensemble.GradientBoostingRegressor,
        {"max_depth": 5,
         "min_samples_leaf": 4,
         "min_samples_split": 7,
         "random_state": 42}
    ],
    "MERGE": [
        AverageTreeRegressor,
        {"ensemble_names": ["Extra Trees", "Random Forest",
                            "Gradient Tree Boosting"],
         "random_states": [42, 42, 42]}
    ],
    "Random Forest": [
        ensemble.RandomForestRegressor,
        {"max_depth": 15,
         "min_samples_leaf": 2,
         "min_samples_split": 7,
         "random_state": 42}
    ],
    "SVR (linear kernel)": [
        svm.LinearSVR,
        {"C": 2,
         "epsilon": 1.0,
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
             "AverageTreeRegressor",
             "DecisionTreeRegressor",
             "ExtraTreesRegressor",
             "GradientBoostingRegressor",
             "RandomForestRegressor"]
