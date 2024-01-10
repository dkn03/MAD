import pandas
import numpy
from matplotlib import pyplot
from scipy.stats import sigmaclip
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder


def pie_diagram(frame, column):
    """
    Отрисовка круговой диаграммы
    :param frame: таблица DataFrame библиотеки pandas
    :param column: название столбца таблицы
    """
    frame.groupby([column]).size().plot(kind='pie')
    pyplot.show()


def bar_diagram(frame, column):
    """
    Отрисовка столбчатой диаграммы
    :param frame: таблица DataFrame библиотеки pandas
    :param column: название столбца таблицы
    """
    frame.groupby([column]).size().plot(kind='bar')
    pyplot.show()


def histogram(frame, column):
    """
    Отрисовка гистограммы
    :param frame: таблица DataFrame библиотеки pandas
    :param column: название столбца таблицы
    """
    frame[column].plot(kind='hist')
    pyplot.show()


def density_plot(frame, column):
    """
    Отрисовка графика плотности
    :param frame: таблица DataFrame библиотеки pandas
    :param column: название столбца таблицы
    """
    frame[column].plot.kde()
    pyplot.show()


def boxplot(frame, column):
    """
    Отрисовка коробчатой диаграммы
    :param frame:
    :param column:
    :return:
    """

    frame.boxplot(column=[column])
    pyplot.show()


def numeric_plot(frame, column):
    """
    Отрисовка графиков для числовых столбцов: гистограммы, графика плотности и коробчатой диаграммы
    :param frame: таблица DataFrame библиотеки pandas
    :param column: название столбца таблицы
    """
    fig, axis = pyplot.subplots(3, 1)
    fig.suptitle(column)
    fig.set_figwidth(20)
    pyplot.subplots_adjust(wspace=0.35)
    pyplot.title(column)
    pyplot.subplot(1,3,1)
    frame[column].plot(kind='hist')
    pyplot.subplot(1,3,2)
    frame[column].plot.kde()
    pyplot.subplot(1,3,3)
    frame.boxplot(column=[column], vert=False)
    pyplot.show()


def categorical_plot(frame, column):
    """
    Отрисовка графиков для категориальных столбов: круговой и столбчатой диаграммы
    :param frame: таблица DataFrame библиотеки pandas
    :param column: название столбца таблицы
    """
    fig, axis = pyplot.subplots(2, 1)
    fig.suptitle(column)
    pyplot.subplots_adjust(wspace=1)
    pyplot.title(column)
    pyplot.subplot(1,2,1)
    frame.groupby([column]).size().plot(kind='pie', title='Круговая диаграмма')
    pyplot.subplot(1,2,2)
    frame.groupby([column]).size().plot(kind='bar',xlabel='Категориальные значения', ylabel='Частота появления в таблице',
                                       title='Столбчатая диаграмма')
    pyplot.show()


def replace_values(series, replaced_values):
    """
    Замена значений в таблице
    :param series: Столбец, в котором заменяются значения
    :param replaced_values: словарь вида {Старое значение : новое значение}
    :return: новый столбец
    """
    return series.map(lambda x: replaced_values[x] if x in replaced_values else x)


def quart_method(frame, column):
    """
    Метод квартилей для столбца
    :param frame: Таблица DataFrame библиотеки Pandas
    :param column: Название столбца таблицы
    :return: Два списка, первый из которых содержит внутренний границы, а второй внешние
    """
    df = frame.sort_values(by=column)
    description = df[column].describe()
    median, low_quartile, high_quartile = description['50%'], description['25%'], description['75%']
    interquartile = high_quartile - low_quartile
    inner_borders = (low_quartile - 1.5 * interquartile, high_quartile + 1.5 * interquartile)
    outer_borders = (low_quartile - 3 * interquartile, high_quartile + 3 * interquartile)
    return inner_borders, outer_borders


def quart_clip(frame, column):
    """
    Удаление выбросов из столбца методом квартилей
    :param frame: Таблица DataFrame библиотеки Pandas
    :param column: Название столбца таблицы
    :return: Новый столбец с удаленными выбросами
    """
    borders = quart_method(frame, column)[0]
    return frame[(frame[column] >= borders[0])
                 & (frame[column] <= borders[1]) | numpy.isnan(frame[column])].reset_index(drop=True)


def sigma_clip(frame, column):
    """
    Удаление выбросов из столбца методом сигм
    :param frame: Таблица DataFrame библиотеки Pandas
    :param column: Название столбца таблицы
    :return: Новый столбец с удаленными выбросами
    """
    clipped = sigmaclip(frame[column].dropna(), 3, 3)
    lower_border, upper_border = clipped[1], clipped[2]
    return frame[(frame[column] >= lower_border)
                            & (frame[column] <= upper_border)
                 | (numpy.isnan(frame[column]))].reset_index(drop=True)


def categorical_knn(frame, column):
    """
    Заполнение пустых значений в категориальном столбце методом k ближайших соседей
    :param frame: Таблица DataFrame библиотеки Pandas
    :param column: Название столбца таблицы
    :return: новая таблица с заполненными значениями
    """
    knn_imputer = KNNImputer()
    encoder = OneHotEncoder()
    if frame[column].isnull().sum() == 0:
        print("There are no empty lines")
        return frame
    frame[column] = frame[column].fillna('empty')
    transformed = encoder.fit_transform(frame[[column]]).toarray()
    columns = encoder.get_feature_names_out()
    new_frame = pandas.DataFrame(transformed, columns=columns)
    new_frame.loc[new_frame[column + '_empty'] == 1] \
        = new_frame.loc[new_frame[column + '_empty'] == 0].replace(0, numpy.nan)
    data_frame = frame.join(new_frame)
    data_frame.iloc[:, 2:] = pandas.DataFrame(knn_imputer.fit_transform(data_frame.iloc[:, 2:]))
    data_frame[column] = encoder.inverse_transform(data_frame[columns])
    return data_frame
