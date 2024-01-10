
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB


def encode_onehot_categorical(frame, columns):
    """
    Кодирует столбцы columns с помощью OneHotEncoder, удаляя первоначальные столбцы
    :param frame: таблица DataFrame
    :param columns: кодируемые столбцы
    :return: новая таблица
    """
    encoder = OneHotEncoder()
    for column in columns:
        transformed = encoder.fit_transform(frame[[column]]).toarray()
        new_frame = pandas.DataFrame(transformed, columns=encoder.get_feature_names_out())
        frame = frame.join(new_frame)

    frame = frame.drop(columns=columns)
    return frame


def encode_label_categorical(frame, columns):
    """
    Кодирует столбцы columns с помощью LabelEncoder, удаляя первоначальные столбцы
    :param frame: таблица DataFrame
    :param columns: кодируемые столбцы
    :return: новая таблица
    """
    encoder = LabelEncoder()
    for column in columns:
        transformed = encoder.fit_transform(frame[column])
        frame[column] = transformed
    return frame


def process_frame(frame, categorical, numeric, drop, encoder='label'):
    """
    Первоначальная обработка таблицы: удаление лишних столбцов, заполнение пропусков, кодирование
    :param frame: Обрабатываемый столбец
    :param categorical: Категориальные столбцы таблицы
    :param numeric: Числовые столбцы таблицы
    :param drop: Удаляемые столбцы
    :param encoder: Метод кодирования категориальных столбцов
    :return: новый обработанный DataFrame
    """
    frame = frame.drop(columns=drop)
    frame[categorical] = frame[categorical].astype('category')
    frame["Age"] = frame["Age"].fillna(frame["Age"].mean())
    frame[numeric] = MinMaxScaler().fit_transform(frame[numeric])
    if encoder =='label':
        return encode_label_categorical(frame, categorical)
    elif encoder == 'onehot':
        return encode_onehot_categorical(frame, categorical)
    else:
        raise ValueError


def create_samples(frame):
    """
    Создает таблицы DataFrame с помощью разных методов сэмплирования
    :param frame: первоначальная таблица
    :return: словарь вида {Вид сэмплирования: таблица}
    """
    samples = {"No_sampling": frame}
    x_resampled, y_resampled = SMOTE().fit_resample(frame.drop(columns='Survived'), frame["Survived"])
    samples['Oversampling'] = x_resampled.join(y_resampled)
    x_resampled, y_resampled = TomekLinks().fit_resample(frame.drop(columns='Survived'), frame['Survived'])
    samples['Undersampling'] = x_resampled.join(y_resampled)
    return samples


def logistic_regression(x_train, y_train):
    """
    Обучение классификатора методом логистической регрессии
    :param x_train: Входные тренировочные значения
    :param y_train: 
    :return:
    """
    parameters = {'C':range(1, 10), 'penalty':(None, 'l1', 'l2', 'elasticnet'), 'solver':(['saga']), 'l1_ratio':([0.5])}
    clf = LogisticRegression()
    grid = GridSearchCV(clf, parameters, scoring='roc_auc')
    grid.fit(x_train, y_train)
    best_clf = grid.best_estimator_
    return best_clf

def kneighbours(x_train, y_train):
    parameters = {'n_neighbors':range(1, 11), 'weights':('uniform', 'distance'),
              'algorithm':('auto', 'ball_tree', 'kd_tree', 'brute')}
    neigh = KNeighborsClassifier()
    grid = GridSearchCV(neigh, parameters, scoring='roc_auc')
    grid.fit(x_train, y_train)
    best_neigh = grid.best_estimator_
    return best_neigh


def gaussian(x_train, y_train):
    gaus_clf = GaussianNB()
    scores = cross_validate(gaus_clf, x_train, y_train, scoring='roc_auc', return_estimator=True)
    c = 0
    for i, x in enumerate(scores['test_score']):
        if x > scores['test_score'][c]:
            c = i
    return scores['estimator'][c]


def discriminant_analysis(x_train, y_train):
    disc_clf = LinearDiscriminantAnalysis()
    scores = cross_validate(disc_clf, x_train, y_train, scoring='roc_auc', return_estimator=True)

    c = 0
    for i, x in enumerate(scores['test_score']):
        if x > scores['test_score'][c]:
            c = i
    return scores['estimator'][c]

def vector_classification(x_train, y_train):
    parameters = {'kernel':('rbf', 'linear', 'poly'), 'C':(range(1, 10))}
    lsvc = SVC(max_iter=1000)

    grid = GridSearchCV(lsvc, parameters, scoring='roc_auc')
    grid.fit(x_train, y_train)
    best_lsvc = grid.best_estimator_
    return best_lsvc