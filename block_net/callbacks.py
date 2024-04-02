import gc                        # очиска памяти
import time                      # библиотека времени
import tensorflow as tf           # библиотека машинного обучения

#Колбек для подсчета в функции ниже средней скорости на эпохе
class TimeHistory(tf.keras.callbacks.Callback):
    # создаем пустой список вначале обучения
    def on_train_begin(self, logs={}):
        self.times = []
    # запоминаем время вначале эпохи
    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()
    # запоминаем время обучения на эпохе
    def on_epoch_end(self, batch, logs={}):
        # добавляем в список в конце эпохи
        self.times.append(time.time() - self.epoch_time_start)

# Custom Callback To Include in Callbacks List At Training Time
class GarbageCollectorCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
