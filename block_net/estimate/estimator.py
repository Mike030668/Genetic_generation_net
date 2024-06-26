from stopit import threading_timeoutable as timeoutable
from block_net.constant import  MESSAGE_2
from block_net.callbacks import  TimeHistory, GarbageCollectorCallback
from block_net.estimate.utils import get_scalepred, auto_corr
import tensorflow as tf           # библиотека машинного обучения
import numpy as np # библиотека нампи
import gc                        # очиска памяти

# Функция на оценки с добавленным колбеком времени
@timeoutable(default = MESSAGE_2) # Декоратор для контроля времени
def evaluate_model(model,                  
                   type_data : str,
                   train_data,
                   val_data,
                   ep,
                   verb,
                   optimizer,
                   loss,
                   channels,
                   predict_lag:int,
                   y_scaler = None,
                   make_log = False,
                   x_val = [],
                   y_val = [],
                   check_aotocorr = False
                   ):
      '''
      Функция оценки модели на точность и автокорреляцию, с обучение
      и проверкой эффекта автокорреляции
      model       - тестируемая модель
      y_scaler    - ранее обученный скэйлер для ответов
      type_data   - "generator" или "numpy"
      type_data   - генератор, или нампи массим (x,y) данных для обучения модели
      val_data    - генератор, или нампи массим (x,y) данных для проверки модели
      ep          - количество эпох оценосного обучения
      verb        - показывать ли процесс обучения
      optimizer   - используемый оптимайзер для обучения
      loss        - используемая функция потерь для обучения
      channels    - каналы в ответе модели для проверки автокорреляции
      predict_lag - на сколько шагом предсказание
      check_aotocorr - проверять автокорреляцию
      '''
      # сбрасываем оценку на случай пересечения названия с global переменной
      val = 0
      # Компилируем модель
      model.compile(optimizer, loss)
      # инициализируем колбек в дальнейшем для поиска более быстрых и оптимизации поиска
      time_callback = TimeHistory()
      # очистка ОЗУ
      clear_ozu = GarbageCollectorCallback()
      # понижение шага
      reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                      mode='min',
                                                      factor = 0.6,
                                                      patience = 1,
                                                      min_lr = 1e-9,
                                                      verbose = 1
                                                      )
      if type_data == "generator":
         # обучаем модель
         history = model.fit(train_data,
                              epochs=ep,
                              verbose=verb,
                              validation_data=val_data,
                              callbacks=[time_callback, clear_ozu, reduce_lr])
        
      elif type_data == "numpy":
        # обучаем модель
         history = model.fit(train_data[0],
                              train_data[1],
                              epochs=ep,
                              verbose=verb,
                              validation_data=val_data,
                              callbacks=[time_callback, clear_ozu, reduce_lr])
        
      else:
         print("type_data can be 'generator' or 'numpy'")

      # получаем данные по времени каждой эпохи
      times_back = time_callback.times
      # берем среднее время эпохи
      time_ep = np.mean(times_back)

     # Считаем MAE автокорреляции и умножаем (прибавляем) ошибку обучения
      val = history.history["val_loss"][-1]

      if check_aotocorr:
         # Прогнозируем данные текущей сетью
         #(pred_val, y_val_true) = get_scalepred(model, XVAL, YVAL, y_scaler)
         (pred_val, y_val_true) = get_scalepred(model, x_val, y_val, y_scaler, make_log)

         # Возвращаем автокорреляцию
         corr, own_corr = auto_corr(pred_lags = channels,
                                    corr_steps = predict_lag,
                                    y_pred = pred_val,
                                    y_true = y_val_true,
                                    show_graf = False,
                                    return_data = True)
      
         val+= tf.keras.losses.MAE(corr, own_corr).numpy()

      # чистим память
      tf.keras.backend.clear_session()
      del model, optimizer, loss
      gc.collect()
      # Возвращаем точность и среднее время эпохи
      return val, time_ep