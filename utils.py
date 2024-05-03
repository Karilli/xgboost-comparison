# NOTE: Import of auto-sklearn version that supports SMOTE, source has to be 
# downloaded localy. Modify the path so that it points to the 
# "/auto-sklearn/autosklearn" folder. You may encounter some problems with 
# requirements.txt and automl_common. Just make sure that they are downloaded
# and you can open them.

if True:
    import sys
    sys.path.insert(0, "../my_autosklearn")


from enum import Enum
from math import floor, sqrt
from xgboost import XGBClassifier, XGBRFRegressor
from autosklearn.pipeline.components.base import AutoSklearnClassificationAlgorithm

from autosklearn.pipeline.constants import SPARSE, DENSE, UNSIGNED_DATA, INPUT
from ConfigSpace.configuration_space import ConfigurationSpace

import autosklearn.pipeline.components.data_preprocessing
import autosklearn.pipeline.components.classification


from autosklearn.classification import AutoSklearnClassifier
from autosklearn.metrics import roc_auc, f1

from sklearn.model_selection import StratifiedKFold
from os.path import exists
import shutil
import json
import os
import pickle
import numpy as np


# NOTE: auto-sklearn might die because it doesnt have enough
# memory, in that case increase MEM to 6k or 9k
# NOTE: u have to directly modify weghting in my_autosklearn
# to enable weights for xgboost. Make sure that "XGBClassifier_"
# is in the list 'clf_' on line 253 in file
# autosklearn/pipeline/compoenents/data_preprocessing/balancing/__init__.py
TIME = 60 * 60
TIME_PER_RUN = 5 * 60
MEM = 3000


class Params:
    BASE_LEARNER = "RF"
    SEED = 0
    K_FOLDS = 5
    VALIDATE = False


TEMP = "../temp_folder"
VALIDATION_DIR = "../results/validation"
MODELS_DIR = "../results/models"


class Task(Enum):
    DATA_IMBALANCE = 1
    FEATURE_INADEQUACY = 2
    SEMI_SUPERVISED = 3
    NOISY_DATA = 4


class LearnerType(Enum):
    CLASSIFICATION = 1
    REGRESSION = 2


def sorted_class_count(y):
    (l1, l2), (c1, c2) = np.unique(y, return_counts=True)
    (c1, l1), (c2, l2) = sorted(((c1, l1), (c2, l2)))
    return c1, c2


def get_XGBModel(X, y, base_learner, task):
    m = X.shape[1]

    if task == LearnerType.CLASSIFICATION:
        XGBModel = XGBClassifier
        objective = "binary:logistic"
    elif task == LearnerType.REGRESSION:
        XGBModel = XGBRFRegressor
        objective = "reg:squarederror"
    else:
        assert False, f"Wrong arguent: {task}."

    if base_learner == "RF":
        return XGBModel(
            learning_rate = 0.2, # 0.0 - 1.0, log, not sure if each forest can have its own, it might lead to overfitting anyway
            max_depth = 20, # 2 - 50, 6 - 20 is another more conservative option
            subsample = 0.63, # 0.0 - 1.0
            colsample_bynode = floor(sqrt(m))/m, # m..num_of_features, 0 - m, log?
            n_estimators = 10, # 100 - 500 = number of random forests in booster
            num_parallel_tree = 10, # 100 - 500 = number of trees in each random forest
            reg_lambda = 0, # -10. - 10.0, log = prunning of trees, higher value -> more prunning, not sure if negative values do anything
            min_child_weight = 2, # 0.0 - 10.0, log, higher value -> less options to choose from when selecting new nodes in trees
            objective = objective, # list at https://xgboost.readthedocs.io/en/stable/parameter.html, search for objective
            seed=Params.SEED,
            seed_per_iteration=True,
            n_jobs=-1,
        )
    elif base_learner == "DecisionTree":
        return XGBModel(
            objective=objective,
            seed=Params.SEED,
            seed_per_iteration=True,
            n_jobs=-1,
        )
    else:
        assert False, f"Wrong arguent: {base_learner}."


class XGBClassifier_(AutoSklearnClassificationAlgorithm):
    def __init__(self, **kwargs):
        self.estimator = None
        for key, val in kwargs.items():
            setattr(self, key, val)

    def fit(self, X, y, sample_weight=None):
        # NOTE: use this assert to validate if weighting is turned on
        # for xgboost, u will see crashed status for configurations 
        # with "balancing:strategy": "weighting"
        # assert sample_weight is None

        self.estimator = get_XGBModel(
            X, y, Params.BASE_LEARNER, LearnerType.CLASSIFICATION
        ).fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        return self.estimator.predict(X)

    def predict_proba(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        return self.estimator.predict_proba(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {
            "shortname": "xgboost",
            "name": "xgboost",
            "handles_regression": False,
            "handles_classification": True,
            "handles_multiclass": True,
            "handles_multilabel": True,
            "handles_multioutput": True,
            "is_deterministic": True,
            "input": (SPARSE, DENSE, UNSIGNED_DATA),
            "output": (INPUT,),
        }

    @staticmethod
    def get_hyperparameter_search_space(feat_type=None, dataset_properties=None):
        return ConfigurationSpace()


def stop_after_100_configurations_callback(smbo, run_info, result, time_left):
    return sum("SUCCESS" in str(val.status) for val in smbo.runhistory.data.values()) <= 100


def get_AutoSklearnClassifier(X, y, task):
    if task == Task.DATA_IMBALANCE:
        metric=[roc_auc]
        include={
            "data_preprocessor": ["feature_type"],
            "balancing": ["none", "weighting", "SVMSMOTE", "ADASYN", "SMOTETomek", "SMOTEENN", "BorderlineSMOTE", ],
            "feature_preprocessor": ["no_preprocessing"],
            "classifier": ["XGBClassifier_"]
        }
    elif task == Task.INADEQUATE_FEATURES:
        metric=[f1]
        include={
            "data_preprocessor": ["feature_type"],
            "balancing": ["none"],
            "classifier": ["XGBClassifier_"]
        }
    else:
        assert False, f"Wrong arguent: {task}."

    if exists(TEMP):
        shutil.rmtree(TEMP, ignore_errors=True)

    return AutoSklearnClassifier(
        time_left_for_this_task=TIME,
        metric=metric,
        initial_configurations_via_metalearning=0,
        ensemble_class=None,
        include=include,
        resampling_strategy=StratifiedKFold(n_splits=Params.K_FOLDS, shuffle=True, random_state=Params.SEED),
        seed=Params.SEED,
        tmp_folder=TEMP,
        delete_tmp_folder_after_terminate=False,
        n_jobs=1,
        memory_limit=MEM,
        get_trials_callback=stop_after_100_configurations_callback,
        per_run_time_limit=TIME_PER_RUN,
    ).fit(X, y)


def get_file_name(root, base_learner, task, dataset_name, ratio=None, ext="json"):
    if ratio is None:
        path = f"{root}/{base_learner}/{str(task)[5:]}/{dataset_name}.{ext}"
    else:
        path = f"{root}/{base_learner}/{str(task)[5:]}/{dataset_name}/{ratio:.2f}.{ext}"
    os.makedirs(os.path.abspath("/".join(path.split("/")[:-1])), exist_ok=True)
    return path


def load_json(base_learner, task, dataset_name, ratio=None):
    try:
        with open(get_file_name(VALIDATION_DIR, base_learner, task, dataset_name, ratio), "r") as f:
            return json.load(f)
    except:
        return {}


def dump_json(dct, base_learner, task, dataset_name, ratio=None):
    def is_jsonable(x):
        try:
            json.dumps(x)
            return True
        except (TypeError, OverflowError):
            return False

    assert is_jsonable(dct), "You have tried to dump object that is not jsonable."

    path = get_file_name(VALIDATION_DIR, base_learner, task, dataset_name, ratio)
    try:
        with open(path, "w") as f:
            json.dump(dct, f)
    except:
        with open(path, "w") as f:
            json.dump(dct, f)


def load_pkl(base_learner, task, dataset_name, ratio=None):
    path = get_file_name(MODELS_DIR, base_learner, task, dataset_name, ratio, "pkl")
    assert exists(path)

    with open(path, "rb") as f:
        return pickle.load(f)


def dump_pkl(model, base_learner, task, dataset_name, ratio=None):
    path = get_file_name(MODELS_DIR, base_learner, task, dataset_name, ratio, "pkl")
    try:
        with open(path, "wb") as f:
            pickle.dump(model, f)
    except:
        with open(path, "wb") as f:
            pickle.dump(model, f)


autosklearn.pipeline.components.classification.add_classifier(XGBClassifier_)