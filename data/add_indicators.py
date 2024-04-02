import numpy as np # библиотека нампи
import pandas as pd # библиотека пандас

def add_sma(df: pd.DataFrame, windows:list, indicators: list):
    """
    data - типа OHCL
    windows - набор окон
    Returns a pd.Series sma.
    """
    copy = df.copy()
    for window in windows:
      for col in indicators:
        copy[f'{col}_SMA_{window}'] = copy[col].rolling(window = window).mean()
    return copy


def add_changes(df: pd.DataFrame, depth: int, indicators: list,
                only_indicators = False, type_change = 'diff'):
    """
    df - DataFrame df типа OHCL
    depth - глубина сбора производных
    indicators - индикаторы для получения данных
    only_indicators - выводить все колонки или только от индикаторов
    type_change - выбор между методами .diff() и .pct_change()
    Returns a DataFrame with the vwap.
    """
    methods = ('diff','pct_change')
    assert type_change in methods, f'В type_change доступо {methods}'
    copy = df.copy() if not only_indicators else df[indicators].copy()
    # Расчет индикаторов относительной доходности
    for i in range(1, depth + 1):
        indicators_changes = [f'{ind}_diff_{i}' for ind in indicators]
        # Относительная доходность в сотых долях за период i шагов назад
        for indicator_change, indicator in zip(indicators_changes, indicators):
          if type_change == 'diff':
            copy[indicator_change] = copy[indicator].diff(periods=i)
          elif type_change == 'pct_change':
            copy[indicator_change] = copy[indicator].pct_change(periods=i)
    return copy


def add_ema(df: pd.DataFrame, windows:list, indicators: list):
    """
    data - типа df.Close
    windows - набор окон
    indicators - индикаторы для получения данных
    Returns a pd.Series ema.
    """
    copy = df.copy()
    for window in windows:
      for col in indicators:
        copy[f'{col}_EMA_{window}'] = copy[col].ewm(span=window).mean()

    return copy

def add_macd(df: pd.DataFrame, indicators: list):
    """
    data - типа DataFrame
    indicators - индикаторы для получения данных
    Returns a pd.Series with the MACD.
    """
    copy = df.copy()
    for col in indicators:
      exp1 = copy[col].ewm(span=12, adjust=False).mean()
      exp2 = copy[col].ewm(span=26, adjust=False).mean()
      copy[f'{col}_MACD'] = exp1 - exp2
    return copy


def add_vwap(df: pd.DataFrame, indicators: list):
    """
    df - DataFrame df типа OHCL
    indicators - индикаторы для получения данных
    Returns a DataFrame with the vwap.
    """
    q = df['Volume'].values
    copy = df.copy()
    for col in indicators:
      p = copy[col].values
      copy[f'{col}_VWAP'] = copy.assign(vwap=(p * q).cumsum() / q.cumsum()).vwap
    return copy

def add_obv(df: pd.DataFrame, indicators: list):
    """
    Добавит OBV осцилятор
    df - DataFrame df типа OHCL
    indicators - индикаторы для получения данных
    Returns a DataFrame with the _OBV.
    """
    copy = df.copy()
    for col in indicators:
    # https://stackoverflow.com/a/66827219
      copy[f'{col}_OBV'] = (np.sign(copy[col].diff()) * copy["Volume"]).fillna(0).cumsum()
    return copy

def data_indicators(df:pd.DataFrame, depth: int, windows:list,
                    use_columns:list):
    """
    Функция добавляющая данные индикаторов
    Returns a pd.Series
    """
    copy = df.copy()
    # набираем мз чего дополняем данные
    copy = add_vwap(copy, indicators = use_columns)
    copy = add_obv(copy, indicators = use_columns)
    copy = add_changes(copy, depth, use_columns, type_change = 'pct_change')
    copy = add_sma(copy, windows,indicators = use_columns)
    copy = add_ema(copy, windows,indicators = use_columns)
    copy = add_macd(copy, indicators = use_columns)
    # оставляем use_columns и их производные
    features = list(filter(lambda x: x.split('_')[0] in use_columns, copy.columns))
    return copy[features]

def num_to_class(df:pd.DataFrame, feature:list, column_prefix: str):
    '''
    добавляет колонки ONEHOT полученне из числоого 1D списка/массива feature
    column_prefix - префикс к колонкам ONEHOT
    '''
    copy = df.copy()
    copy[column_prefix] = feature
    copy = pd.get_dummies(copy,
                          columns = [column_prefix],
                          drop_first = True,
                          )
    return copy

def add_from_datetime(df:pd.DataFrame, add_classdays = False):
    """
    Функция добавляет:
    - миксовое числовое представление даты из datetime
    - представление ряда параметров времени как onehot
    Returns a pd.Series
    """
    copy = df.copy()
    # делаем еще колонки на основе даты
    day  =  copy.index.day                   # день месяца
    week_day =  copy.index.dayofweek+1         # день недели
    week  = copy.index.isocalendar()['week']   # недели года
    month = copy.index.month                   # месяц
    year  =  copy.index.year                   # год
    dayofyear =copy.index.dayofyear            # день года
    mix_day = day*week_day*month*year*dayofyear*week
    copy['Mix_day'] = np.log(mix_day).to_numpy()
    if add_classdays:
        copy = num_to_class(copy, week_day, 'cls_dw')
        copy = num_to_class(copy, month, 'cls_my')
    return copy
