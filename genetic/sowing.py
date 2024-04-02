import numpy as np # библиотека нампи
from block_net.constructor import WildregressModel
from block_net.bricks import Set_net, Make_blocks

# функция для ввода ботов и состава блоков моделей нейронок
def posev_net():
    '''
    Функция для передачи посева вначале генетического отбора вручную.
    Вначале нужно будет ввести число сеемых сетей,
    если 0, то код пойдет далее с 0,
    если 1, 2 или n натуральное число, то это кол-во сеемых сетей.
    нужно будет передать для каждой сети свой набор:
    [bot_pop] - бот популяции вида [7, 1, 0, 1, 16, 7, 0, 0, 0, 0]
    [blockov_list] - список слоев блоков вида [[], [], ['Conv1DT'], ['Dense'], [], [], []]
    [bot] - список параметров слоев блоков вида [[], [], [32, 5], [256], [], [], []]

    '''
    nets = int(input('Укажите количество сетей для посева: '))
    posev = []
    for i in range(nets):
        net=[]
        botpop_net  = eval(input('Введите лист бота популяции сети:'))
        net.append(botpop_net)
        blockovlst_net = eval(input('Введите лист блоков сети:'))
        net.append(blockovlst_net)
        bot_net = eval(input('Введите бот_лист параметров слоев сети:'))
        net.append(bot_net)
        posev.append(net)
    return posev

# функция определения индексов лучших
def get_idxbest(sval: list, best: int):
    '''
    Функция для получения индексов и точностей
    из матрицы
    Args:
      sval -  сортированная матрицы оценок
      int - нужно количество лучших
    Return:
      idxs - индексы лучших
      sval_best - оценки лучших
    '''
    flt =  np.array(sval).ravel()  # вытягиваем массив
    flt = np.sort(flt)# сортируем слева направо
    # ищем индексы best лучших
    idxs = np.array([np.array(np.where(sval == flt[i])).ravel()
                     for i in range(best)])
    sval_best = flt[:best]
    return idxs, sval_best

# функция получения списка лучших сетей
def get_bestnets(idxs, botpop_lst, mega_info, mega_popul):
    '''
    Функция для получения списка лучших сетей для подсева
    в процессе генетического отбора.
    из матрицы
    Args:
      botpop_lst - список списков ботов самих популяций
      mega_info  - список списков имен слоев всех сетей всех популяций
      mega_popul - список списков парамметров слоев всех сетей всех популяций
    Return:
      thebestnets - лучших сетей взятых по индексам idxs
    '''
    thebestnets = []
    for id in idxs:
        net=[]
        i = id[0]
        j = id[1]
        botpop_net  = botpop_lst[i]
        net.append(botpop_net)
        blockovlst_net = mega_info[i]
        net.append(blockovlst_net)
        bot_net = mega_popul[i][j]
        net.append(bot_net)
        thebestnets.append(net)
    return thebestnets

def getnetfrombest(inshape: tuple,
                   thebestnets: list,
                   activ_lays: list,
                   activ_out:list,
                   neiro_out: int,
                   limit = 10,
                   n = 1
                   ):
    '''
    Функция поулченмя смискм из моделей, которые генерируются
    из списка списков парамметров сетей из генгетического отбора
    Args:
      thebestnets - список списков списков по структуре отобранных сетей тип:
                [[[bot_pop_1],[blockov_list_1],[bot_1]],
                [[bot_pop_1],[blockov_list_1],[bot_1]]]
      activ_lays - список возможных активац. функций внутри сети
      activ_out - список возможных активац. функций выхода сети
      neiro_out - кол-во выходных нейронов
      limit - лимит на сборку модели
      n - количество выводимых сетей
    Return:
       modlbest_lst - список сгенерированных сетей длины n
    '''
    modlbest_lst = []
    for i in range(n):
      # определяем тип создания модели
      choosed_net = Set_net(thebestnets[i][0][2], activ_lays,
                            activ_out, neiro_out)
      # инициализируем класс структуры блоков на основе парраметров сети
      maker_blocks = Make_blocks(choosed_net)
      # инициализируем класс формирования сети
      #make_model = WildregressModel(INSHAPE)
      make_model = WildregressModel(inshape)

      model_best = make_model(thebestnets[i][0], # бот_популяции сетей
                              thebestnets[i][2],  # бот парамметров слоев сети
                              thebestnets[i][1] , # список имен слоев сети
                              maker_blocks, # класс построения блоков
                              # парамметр от декаратора
                              timeout = limit,
                              # время в сек отводимое на оценку
                              )
      modlbest_lst.append(model_best)
      
    return modlbest_lst