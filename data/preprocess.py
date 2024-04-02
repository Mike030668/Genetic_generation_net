import numpy as np # библиотека нампи
import pandas as pd # библиотека пандас

def clean_dataset(df:pd.DataFrame):
    """
    data - типа OHCL
    функция очистки
    """
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    # индексы без nan, inf и -inf
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(axis=1)
    return df[indices_to_keep].astype(np.float64)


# Функция разделения набора данных на выборки для обучения нейросети
# x_data - набор входных данных
# predict_lag - количество шагов в будущее для предсказания
def future_sequence(x_data, predict_lag):
    # Определение максимального индекса
    y_len = x_data.shape[0] - (predict_lag - 1)
    # отстоящих на predict_lag шагов вперед
    y = [x_data[i:i+ predict_lag] for i in range(y_len)]
    # Возврат результатов в виде массивов numpy
    return np.array(y)


def df_for_mpf(df: pd.DataFrame, raw_df: pd.DataFrame, columns:list):
  '''
  Добавление колонок Open, High, Low, Close из raw_df
  необходимых для отображения в mpf.plot()
  если удалили для тренировачного набора
  '''
  copy = df.copy()
  for col in columns:
    copy = copy.join(raw_df[col])
  return copy