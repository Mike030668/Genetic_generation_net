import numpy as np              # библиотека нампи
import matplotlib.pyplot as plt # библиотека матплотлиб для отрисовки
import gc                        # очиска памяти

# Вспомогательные функции

def show_pocess(avlmdl_lst: list, gdmdl_lst: list,
                  sval_lst: list, seff_lst: list):
    '''
    Функция отображает процесс создания моделей и
    точность с эффективностью лучшей
    Args:
      avlmdl_lst - Доля созданных моделей
      gdmdl_lst - Доля пригодных моделей
      sval_lst - Ошибка лучшей в популяции
      seff_lst - Эффективность лучшей в популяции
    '''
    plt.figure(1, figsize=(18,6))
    plt.subplot(1,2,1)
    plt.title('Динамика создания моделей')
    plt.plot(avlmdl_lst, label = 'Доля созданных моделей')
    plt.plot(gdmdl_lst, label = 'Доля пригодных моделей')
    plt.legend()
    plt.grid()

    plt.subplot(1,2,2)
    plt.title('Динамика поиска моделей')
    if len(sval_lst) > 15:
        plt.plot(sval_lst[-15:], label = 'Ошибка лучшей в популяции')
        plt.plot(seff_lst[-15:], label = 'Эффективность лучшей в популяции')
    else:
        plt.plot(sval_lst, label = 'Ошибка лучшей в популяции')
        plt.plot(seff_lst, label = 'Эффективность лучшей в популяции')
    plt.legend()
    plt.grid()
    plt.show()



def show_process(svalp_lst:list, seffp_lst:list, ephtime_lst:list):
    '''
    Функция динамику поиска лучшей модели от эпохи
    Args:
      svalp_lst - Ошибка лучшей модели
      seffp_lst - Эффективность лучшей модели
    '''
    plt.figure(1, figsize=(18,6))
    plt.subplot(1,2,1)
    plt.title('Динамика поиска лучшей модели  от эпохи')
    if len(svalp_lst) > 50:
        plt.plot(svalp_lst[-50:], label = 'Ошибка лучшей модели')
        plt.plot(seffp_lst[-50:], label = 'Эффективность лучшей модели')
    else:
        plt.plot(svalp_lst, label = 'Ошибка лучшей модели')
        plt.plot(seffp_lst, label = 'Эффективность лучшей модели')
    plt.legend()
    plt.grid()

    plt.subplot(1,2,2)
    plt.title('Время поиска от эпохи')
    plt.plot(ephtime_lst)
    plt.grid()
    plt.show()
    print()


def get_var_name(variable, globals_dict):
    '''
    Функция получения имени переменной в виде строки
    на основе глобальной видимости
    пример https://www.programiz.com/python-programming/methods/built-in/globals
    variable - переменная
    '''
    #globals_dict = globals()
    var_name = [var_name for var_name in globals_dict
                if globals_dict[var_name] is variable
                  ][0]
    return var_name

def saver(lists_datas: list, directory: str):
    '''
    Функция сохраняет список с именем переменной
    в списке на основы функции, выдающей имя переменной
    get_var_name
      Args:
      lists_datas - список из сохраняемых списков
      directory - директория, куда пишем
    '''
    for data in lists_datas:
        name = get_var_name(data)
        np.save(directory + name + '.npy', np.array(data, dtype=object))
        # удаляем модель
    del(lists_datas)
    # чистим память
    gc.collect()