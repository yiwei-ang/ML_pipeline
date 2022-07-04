from enum import Enum
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR


class ClassificationModels(Enum):
    logistic_regression = (1, LogisticRegression(multi_class='multinomial'))
    linear_discriminant_analysis = (2, LinearDiscriminantAnalysis())
    xgboost = (3, XGBClassifier())
    k_nearest_neighbour = (4, KNeighborsClassifier(5))
    decision_tree = (5, DecisionTreeClassifier())
    random_forest = (6,  RandomForestClassifier())
    adaptive_booster = (7, AdaBoostClassifier())
    support_vector = (8, SVC())

    def __init__(self, id, func):
        self.id = id
        self.func = func


class RegressionModels(Enum):
    gradient_boosting = (1, GradientBoostingRegressor())
    xgboost = (2, XGBRegressor())
    support_vector = (3, SVR())
    catboost = (4, CatBoostRegressor())
    sgd = (5, SGDRegressor())
    elasticnet = (6, ElasticNet())

    def __init__(self, id, func):
        self.id = id
        self.func = func