import pandas
import numpy
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import sigmaclip, t
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, Lars, BayesianRidge, TweedieRegressor
from sklearn.metrics import roc_auc_score, mean_squared_error, r2_score
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV, StratifiedKFold, KFold
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.kernel_ridge import KernelRidge
from statsmodels.nonparametric.kernel_regression import KernelReg
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

def sigma_clip(frame, column):
    """
    Удаление выбросов из столбца методом сигм
    :param frame: Таблица DataFrame библиотеки Pandas
    :param column: Название столбца таблицы
    :return: Новый столбец с удаленными выбросами
    """
    clipped = sigmaclip(frame[column].dropna())
    lower_border, upper_border = clipped[1], clipped[2]
    return frame[(frame[column] >= lower_border)
                 & (frame[column] <= upper_border)
                 | (numpy.isnan(frame[column]))].reset_index(drop=True)


def create_heatmap(corr_df):
    sns.set(rc={"figure.figsize": (30, 30)})
    sns.heatmap(corr_df, annot=True)
    # pandas.set_option('display.max_rows', None)
    # pandas.set_option('display.max_columns', None)


def get_model_coef_stats(model, coef, X, y, i):
    # Вычисление стандартной ошибки
    y_pred = model.predict(X)
    mse = ((y_pred - y) ** 2).mean()
    s_err = np.sqrt(mse * np.linalg.inv(X.T @ X).diagonal())[i]
    # Статистика Стьюдента
    t_value = coef / s_err
    # Проверка значимости коэффициента
    df = X.shape[0] - X.shape[1]
    p_value = 2 * (1 - t.cdf(np.abs(t_value), df))
    is_valuable = (np.abs(t_value) > 1.96) & (p_value < 0.05)
    # Вычисление 95%-доверительного интервала
    se = s_err * t.ppf(0.95, df)
    lower_ci = coef - se
    upper_ci = coef + se
    return t_value, p_value, is_valuable, lower_ci, upper_ci


def get_model_stats(model, X_test, y_test):
    if hasattr(model, "predict"):
        y_pred = model.predict(X_test)
    else:
        y_pred = model.fit(X_test)[0]
    n, p = X_test.shape
    r2 = r2_score(y_test, y_pred)
    adj_r2 = 1 - ((1 - r2) * (n - 1)) / (n - p - 1)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    aic = n * np.log(mean_squared_error(y_test, y_pred)) + 2 * p
    bic = n * np.log(mean_squared_error(y_test, y_pred)) + p * np.log(n)
    res = pandas.Series({"R^2": r2, "Adj r^2": adj_r2, "RMSE": rmse, "AIC": aic, "BIC": bic})
    return res


def coef_stats(model, X, y, lib=None):
    coef_data = {'Name': [], 't-value': [], 'p-value': [], 'Is valuable': [],
                 '95% Confidence Interval': [], }
    if lib == 'scl':
        coeffs = model.coef_
    else:
        coeffs = model.params.values
    for i, col_name in enumerate(X.columns):
        res = get_model_coef_stats(model, coeffs[i], X, y, i)
        coef_data['Name'].append(col_name)
        coef_data['t-value'].append(res[0])
        coef_data['p-value'].append(res[1])
        coef_data['Is valuable'].append(res[2])
        coef_data['95% Confidence Interval'].append(f'[{res[3]}, {res[4]}]')
    return pandas.DataFrame(coef_data)
