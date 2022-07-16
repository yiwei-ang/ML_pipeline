from engine.util.supervised_model import ClassificationModels, RegressionModels

from scipy.stats import skew, boxcox
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt
from typing import Dict

MAPPING = {
    'classification': ClassificationModels,
    'regression': RegressionModels
}


def standardize_input_for_training(df: pd.DataFrame):
    # Assumed the rightmost column is label/target
    last_column = df.columns[-1]
    if 'y' not in [name.lower() for name in df.columns]:
        df['y'] = df[df.columns[-1]]
        df.drop(columns=last_column, axis=1, inplace=True)

    # We need the column y (label/target) to be readable by numpy which is in integer.
    label_mapping = {}
    le = LabelEncoder()
    if df['y'].dtype.kind != 'i':
        df['y'] = le.fit_transform(np.array(df[['y']]).ravel())
        label_mapping = dict(zip(le.classes_, range(len(le.classes_))))
    return df, list(df.columns[:-1]), last_column, label_mapping


def apply_simple_imputer_and_encoding(df: pd.DataFrame, strategy='mean'):
    # create two DataFrames, one for each data type
    df_numeric, df_categorical = [df[df.select_dtypes(i).columns] for i in ['number', 'object']]

    # Apply SimpleImputer to numeric columns
    imp = SimpleImputer(missing_values=np.nan, strategy=strategy)
    df_numeric = pd.DataFrame(imp.fit_transform(df_numeric), columns=df_numeric.columns)

    # Label Encoder for categorical data
    if len(df_categorical.columns) > 0:
        df_categorical = pd.DataFrame(LabelEncoder().fit_transform(df_categorical), columns=df_categorical.columns)

    df = pd.concat([df_numeric, df_categorical], axis=1)

    return df


def apply_boxcox_transformation(df: pd.DataFrame, threshold=0.7):
    # Do for all
    numerical_features = list(df.select_dtypes('number').columns)
    skewed_features = df[numerical_features].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

    # compute skewness
    skewness = pd.DataFrame({'skew': skewed_features})

    # Get only highest skewed features
    skewness = skewness[abs(skewness) > threshold]
    skewness = skewness.dropna()

    fitted_lambdas = {}

    for feat in skewness.index:
        df[feat], fitted_lambdas[feat] = boxcox((df[feat] + 1))
    return df, fitted_lambdas


def prepare_input_for_training(df: pd.DataFrame, test_size=0.2):
    df = apply_simple_imputer_and_encoding(df)
    df = apply_boxcox_transformation(df)[0]

    x = df[[i for i in df.columns if i not in ['y']]]
    y = df['y']

    return train_test_split(x.to_numpy(), y.to_numpy(), test_size=test_size)


def correlation_matrix(df):
    fig, ax = plt.subplots(1, 1, figsize=(20, 8))
    corr = df.corr()
    ax.set_title("Correlation Matrix")
    top_corr_cols = corr.quality.sort_values(ascending=False).keys()
    top_corr = corr.loc[top_corr_cols, top_corr_cols]
    dropSelf = np.zeros_like(top_corr)
    dropSelf[np.triu_indices_from(dropSelf)] = True
    return sns.heatmap(top_corr, cmap=sns.diverging_palette(220, 10, as_cmap=True), annot=True, fmt=".2f",
                       mask=dropSelf, ax=ax)


def choose_best_model(data: Dict, scoring='mean'):
    df = pd.DataFrame(data)
    df = df.sort_values(scoring, ascending=False)
    return df.iloc[0].model_name


class SupervisedModels():
    def __init__(self, input_data, problem_type='classification', evaluation_metric='accuracy'):
        self.input_data, self.features, self.label, self.label_mapping = standardize_input_for_training(input_data)
        self.problem_type = problem_type
        self.models = MAPPING.get(self.problem_type)
        self.evaluation_metric = evaluation_metric

    def _confusion_matrix(self, y_test, predictions):
        unique_y = list(np.sort(np.unique(np.concatenate((y_test, predictions), axis=None))))
        print(unique_y)
        print(confusion_matrix(y_test, predictions))
        conf_matrix_raw = confusion_matrix(y_test, predictions)
        conf_matrix = pd.DataFrame(confusion_matrix(y_test, predictions),
                                   columns=unique_y,
                                   index=unique_y)
        conf_matrix.index.name = 'Actual'
        conf_matrix.columns.name = 'Predicted'
        conf_matrix = conf_matrix.unstack().rename('value').reset_index()

        return conf_matrix.to_dict(orient='records'), conf_matrix_raw

    def model_fitting(self, x_train, x_test, y_train, y_test, results_k_fold, model='random_forest'):
        selected_model = self.models[model].func
        selected_model.fit(x_train, y_train)
        predictions = selected_model.predict(x_test)

        score = accuracy_score(y_test, predictions)
        recall = recall_score(y_test, predictions, average='weighted')
        precision = precision_score(y_test, predictions, average='weighted')
        f1_score_ = f1_score(y_test, predictions, average='weighted')
        conf_matrix = self._confusion_matrix(y_test=y_test, predictions=predictions)

        output = {
            "model_name": model,
            "features": self.features,
            "labels": self.label,
            "label_mapping": list(self.label_mapping.keys()),
            "metrics": {
                "accuracy_score": score,
                "recall": recall,
                "precision": precision,
                "f1_score": f1_score_},
            "confusion_matrix": conf_matrix[1],
            "results_k_fold": results_k_fold
        }
        return output

    def run_pipeline(self):
        X_train, X_test, y_train, y_test = prepare_input_for_training(self.input_data)
        models = list(self.models.__members__.keys())
        summary_k_fold = []
        results_k_fold = []
        for model in models:
            k_fold = StratifiedKFold(n_splits=5)
            cv_results = cross_val_score(self.models[model].func, X_train, y_train, cv=k_fold, scoring='accuracy')
            results_k_fold.append({'model_name': model, 'cv_values': list(cv_results)})
            summary_k_fold.append({'model_name': model, 'mean': cv_results.mean(), 'std': cv_results.std()})

        # Use the best model to provide analysis to the users.
        best_model = choose_best_model(summary_k_fold)

        return self.model_fitting(X_train, X_test, y_train, y_test, model=best_model, results_k_fold=results_k_fold)
