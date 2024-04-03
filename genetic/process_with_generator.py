import gc                        # очиска памяти
import time                      # библиотека времени
import random
import numpy as np              # библиотека нампи
from tqdm.notebook import tqdm   # отрисовка прохождения цикла
import tensorflow as tf           # библиотека машинного обучения
from IPython.display import clear_output # очистка вывода в ячейке
import warnings # библиотека сообщений по ошибкам
warnings.filterwarnings("ignore") # игнорировать сообщения ошибок

from genetic.sowing import posev_net, get_idxbest, get_bestnets
from genetic.utils import show_pocess, show_process, saver
from block_net.bricks import Set_net, Make_blocks
from block_net.constructor import WildregressModel
from block_net.constant import  TIMELIMIT_1, TIMELIMIT_2
from block_net.estimate.estimator import  evaluate_model

def selection( 
        directory: str, # куда  пишет данные этот код
        waitnets: int,
        dw: float, #
        frbest: int,
        inshape: tuple,
        predit_lag: int, 
        make_log: bool,
        x_val: list,
        y_val: list,
        type_data : str,
        train_data: object,       # генератор данных для обучения
        val_data: object,
        y_scaler: object,         # Y_SCAILER,        # обученный скейлер для y
        activ_lays: list,
        neiro_out: int,
        activ_out: str,
        style_net: dict,
        q_tyblocks: list,
        q_lays: list,
        verbouse: int,
        epohs: int,
        test_eph: int,
        optimizer:object,
        loss:object,
        n: int,                   # = 5 # количество ботов популяции
        p: int,                   #  = 6  # количество популяций
        dn: float,                # = 0.3 # доля выживших ботов
        dp: float,                # = 0.3 # доля выживших популяций
        dneff: float,             # = 0.1 # доля выживших ботов по эффективности
        dpeff: float,             # = 0.1 # доля выживших популяций по эффективности
        prb_randbot: float,       # = 0.3 # вероятность появления случайного бота в новой популяции
        mutp: float,              # = 0.4   # Коэфициент мутаций при создании мегабота новой популяции
        mutn: float,              # = 0.45  # Коэфициент мутаций при создании бота новой сети в популяции
        dpsurv: float,            # = 0.8 # доля от выживших ботов популяции используемыех в родителях
        dnsurv: float,            # = 0.8 # доля от выживших ботов мегапопуляции используемыех в родителях
        posev = [],
        check_aotocorr = True
):
        """
         directory -  куда  пишет данные этот код
        ########### посев сетей вначале кода ############################################
        # posev = []
        # смотри пояснения выше как передать сюда ранее созданные данные
        # posev = np.load(directory +'bestnets.npy', allow_pickle = True)
        # если еще нет список моделий, то ввести 0
        # или можно вручную списки подать, указав вместо 0, сколько будкт сетей
        # последовательно в ответ ввести  листы [bot_pop], [blockov_list], [bot]

        # в текстовой ячейке ниже есть 2 сети для посева вручную

        ########### для подсевания лучших сетей прошлых эпох ###########################
        waitnets = 3 # сколько выводим лучших для для изучения и посева
        dw = 0.4 # доля лучших с прошлых эпох для подсева
        frbest = 2 # как часто подсевыем лучших с прошлых эпох

        ########## папаметпы для генерируемых сетей ###################################
        activ_lays =['relu', 'elu', 'tanh', 'sigmoid', 'selu', 'softmax',
                    'softplus', 'softsign', 'hard_sigmoid', 'exponential']
        # нужное количество входных нейроной
        neiro_out = y_train.shape[1]

        # функции активации для выходного слоя
        activ_out = ['linear','relu', 'elu']
        # словарь типов сетей
        style_net = {0: 'Dense',
                    1: 'Conv',
                    #2: 'Recur', # На малом ОЗУ лучше закомитить
                    None: 'MIX'
                    }

        ################################################################################
        q_tyblocks = 8    # макс количество генерируемых блоков для сети
        q_lays = 10       # макс количество слоев в блоках
        ################################################################################
        verbouse = 0     # отображать ли обучение
        epohs = 3        # Количество эпох для генетического поиска моделей
        test_eph = 5     # Количество эпох тестового обучения моделей
        ################################################################################

        ################################################################################
        n = 5             # количество ботов популяции
        p = 6             # количество популяций

        dn = 0.3 # доля выживших ботов
        dp = 0.3 # доля выживших популяций

        dneff = 0.1 # доля выживших ботов по эффективности
        dpeff = 0.1 # доля выживших популяций по эффективности

        prb_randbot = 0.3 # вероятность появления случайного бота в новой популяции
        mutp = 0.4   # Коэфициент мутаций при создании мегабота новой популяции
        mutn = 0.45  # Коэфициент мутаций при создании бота новой сети в популяции

        dpsurv = 0.8 # доля от выживших ботов популяции используемыех в родителях
        dnsurv = 0.8 # доля от выживших ботов мегапопуляции используемыех в родителях
        ################################################################################
                    
        """
    
        global globals_dict, mega_popul, botpop_lst, mega_info, svalp_lst, ephtime_lst
        global newmega_popul, newbotpop_lst, newmega_info
        globals_dict = globals()


        nsurv = max(2,round(dn* n))  # Кол-во выживших ботов мин 2 для родителей
        psurv = max(2,round(dp * p)) # Кол-во выживших популяций

        nsurv_eff = max(1,round(dneff * n)) # Коли-во выживших ботов поп-ии по эффект.
        psurv_eff = max(1,round(dpeff * p)) # Кол-во выживших ботов мегапоп по эффект.

        nnew = max(0, n - nsurv - nsurv_eff)  # Кол-во новых ботов
        pnew = max(0, p - psurv - psurv_eff)  # Кол-во новых популяций

        parents_n = round(nsurv*dnsurv) # выж. боты популяции используемые в родителях
        parents_p = round(psurv*dpsurv) # выж. боты мегапопуляции используемые в родит.

        sevbest = round(waitnets*dw) # сколько подсеем лучших с прошлых эпох
        ################################################################################


        ################################################################################
        popul = []         # обнулении популяции
        val_p = []         # обнулении точности популяции
        # создаем мегопопуляцию популяций ботов
        mega_popul = []
        mega_info = []
        botpop_lst = []

        start_time = time.time()

        # для посева из определенных моделей нужно создвть лист сетей попримеру
        # или ввести 0 при запросе input()
        if posev == []: posev = posev_net()

        for m in range(p):
        ############# ПОСЕВ ############################################################
            # сеем популяции из листа posev
            if posev != [] and m < len(posev):
                bot_pop = posev[m][0]
                blockov_list = posev[m][1]
                popul = []                 # Создаем пустую популяцию
                popul.append(posev[m][2])  # задаем первого бота популяции из сева
                for i in range(1,n):       # Проходим по всей длине популяции от 1
                    # создаем оставшихся случайнонных ботов из сеяного blockov_list
                    # определяем тип создания модели
                    choosing_net = Set_net(bot_pop[2], activ_lays, activ_out, neiro_out)
                    # инициализируем класс структуры блоков на основе парраметров сети
                    maker_blocks = Make_blocks(choosing_net)
                    bot = maker_blocks.buildblock_bot(blockov_list)
                    popul.append(bot)      # добавляем бота в популяцию

            ############################################################################
            # генерируем бота популяции
            else: # если нет посева или если посев менее нужного количества
                # случайно выбираем тип популяции
                type_net =  random.choice(list(style_net.keys()))#
                # инициализируем Set_net на основе заданных парамметров
                choosing_net = Set_net(type_net, activ_lays, activ_out, neiro_out)
                # инициализируем класс структуры блоков на основе парраметров сети
                maker_blocks = Make_blocks(choosing_net)
                bot_pop = maker_blocks.buildpopulbot(q_tyblocks, q_lays)

                # генерируем из состав блоков из к-ва блоков и слоев
                structure = [np.random.randint(0,bot_pop[1]) for i in range(bot_pop[0])]
                # создаем единый список блоков для популяции
                blockov_list = maker_blocks.sostav_blockov(structure)
                popul = []             # Создаем пустую популяцию
                for i in range(n):     # Проходим по всей длине популяции
                    # создаем очередного случайнонного бота на основе blockov_list
                    bot = maker_blocks.buildblock_bot(blockov_list)
                    popul.append(bot)  # добавляем бота в популяцию

            mega_popul.append(popul) # доб. популяцию в мегапопуляцию
            mega_info.append(blockov_list) # доб. информацию о блоках  в мегапопуляцию
            botpop_lst.append(bot_pop) # доб. мегабота популяции в лист ботов популяций

        if posev != []: print(f'Посеено {len(posev)} сетей')

        # счетчики
        avl_mdl = 0 # счетчик созданных моделей
        non_mdl = 0 # счетчик не созданных моделей
        ntk_mdl = 0 # счетчик моделей не подходящих под задачу
        gd_mdl  = 0 # счетчик моделей пригодных под задачу
        ################################################################################

        ################################################################################
        # для сбора данных об эффективности обучения от эпохи
        ephtime_lst = []
        avlmdl_lst  = []
        gdmdl_lst   = []
        sval_lst    = []
        seff_lst    = []
        svalp_lst   = []
        seffp_lst   = []
        bestnets = []
        val_best = []
        # счетчик ошибок в скрещивании
        err_gen = 0
        ################################################################################

        # Пробегаемся по всем эпохам
        for it in tqdm(range(epohs), unit ="epohs",  # Пробегаемся по всем эпохам
                        desc ="Пробегаемся по всем эпохам"):
            val_p = []
            eff_p = []
            raw_val_p = []
            raw_eff_p = []
            curr_time = time.time()
            for m in tqdm(range(p), unit ="popul",
                        desc ="Проходимся по популяциям"): # проходимся по популяциям
                popul = mega_popul[m]       # берем очередную популяцию
                blockov_list = mega_info[m] # берем информацию и популяции
                bot_pop = botpop_lst[m]     # берем очередного мегабота популяцию
                val = []
                eff = [] #  список для списков по среднему обучению модели в fit()
                for i in range(n): # Проходим по всей длине популяции
                    bot = popul[i] # Берем очередного бота

                    ########################################
                    # определяем тип создания модели
                    choosing_net = Set_net(bot_pop[2], activ_lays, activ_out, neiro_out)
                    # инициализируем класс структуры блоков на основе парраметров сети
                    maker_blocks = Make_blocks(choosing_net)
                    # инициализируем класс формирования сети
                    #regress_model = WildregressModel(INSHAPE)
                    regress_model = WildregressModel(inshape)
                    # тип модели и номер для отображения
                    discription = f'{style_net[bot_pop[2]]}_{i}'
                    ###########################################

                    #####################################################################
                    #               ОЦЕНКА МОДЕЛИ ОТ БОТА ПОПУЛЯЦИИ                     #
                    #####################################################################
                    testing = False
                    try:
                        # пробуем создать модель
                        gen_model = regress_model(
                                                    bot_pop,      # бот_популяции сетей
                                                    bot,          # бот парам-в слоев сети
                                                    blockov_list, # список имен слоев сети
                                                    maker_blocks, # класс построения блоков
                                                    # парамметр декоратора
                                                    timeout = TIMELIMIT_1,
                                                    # время в сек отводимое на создание модели
                                                    )
                        # если превысили время, то gen_model - просто сообщение
                        if type(gen_model) == str:
                            print(gen_model)
                            testing = False
                        else: # значит модель создалась
                            train_param = np.sum([tf.keras.backend.count_params(w) \
                                                    for w in gen_model.trainable_weights])
                            print(discription + f' c {train_param} обучаемыми параметрами - создалась')
                            testing = True
                            avl_mdl+=1

                    except Exception:
                        testing = False
                        # если не создалась то пишем плохую точность
                        print(discription + ' - не создалась')
                        non_mdl+=1
                        f = 1000
                        tlrn = 1000

                    # Вычисляем точность текущего бота
                    try:
                    #if testing:
                        if testing:
                            # оптимизатор
                            #optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
                            # функция потерь
                            #loss = tf.keras.losses.MSE
                            # оценка по времени и смешанной точности нашей модели
                            result = evaluate_model(
                                                    # парамметр от декоратора
                                                    timeout =  TIMELIMIT_2,            # время в сек отводимое на оценку
                                                    # собственные парамметры функции
                                                    model = gen_model,                 # тестируемая модель
                                                    y_scaler = y_scaler,               # обученный скейлер для y
                                                    make_log = make_log,
                                                    x_val = x_val,
                                                    y_val = y_val,
                                                    type_data = type_data,
                                                    train_data = train_data,           # генератор данных для обучения
                                                    val_data = val_data,               # генератор данных для проверки
                                                    ep = test_eph,                     # эпох обучения
                                                    verb = verbouse,                   # отображать ли обучение
                                                    optimizer = optimizer,             # оптимизатор
                                                    loss = loss,                       # функция потерь
                                                    channels = np.arange(predit_lag),  # P REDICT_LAG),# Отображение сводки модели
                                                    predict_lag = predit_lag ,         #    PREDICT_LAG    # На сколько шагов предсказание
                                                    check_aotocorr = check_aotocorr
                                                    )

                            # выводим результат оценки
                            # если превысили время, то gen_model - просто сообщение
                            if len(result) > 2:
                                print(result)
                                ntk_mdl+=1
                                f = 300
                                tlrn = 300

                            else: # значит модель протестировалась
                                f = result[0]
                                tlrn = result[1]
                                print(discription + ' - подошла под задачу')
                                gd_mdl+=1

                            # удаляем модель
                            del(gen_model)
                            # чистим память
                            gc.collect()

                        else:
                            print(discription + ' - слишком долго создавалась')
                            ntk_mdl+=1
                            f = 600
                            tlrn = 600

                    except Exception:
                    #else:        
                    # если не создалась то пишем плохую точность
                        print(discription + ' - не подошла под задачу')
                        ntk_mdl+=1
                        f = 800
                        tlrn = 800


                    if f in (600, 800, 1000): print('Модель отбракована')
                    elif f == 300: print('Модель долго учится')
                    else:
                        print(f'Оценка модели {round(f, 5)}, тестовое обучение на {test_eph} эпохах составило {round(tlrn, 2)}сек. ')
                    print()
                    val.append(f)       # Добавляем полученное значение в список val
                    eff.append(tlrn*f) # сохраняем время эффективность обучения модели
                    #####################################################################

                ########################################################################
                # базовый список точновстей мегапопуляции
                raw_val_p.append(val)
                # сортируем val
                sval = sorted(val, reverse=0)
                # базовый список сортированных точновстей популяций
                val_p.append(sval)
                # для сбора динамики точности от популяции
                sval_lst.append(np.log10(sval[0]))


                # базовый список эффекивностей мегапопуляции
                raw_eff_p.append(eff)
                # сортируем по эффективности
                seff = sorted(eff, reverse=0)
                # базовый список сортированных эффекивностей популяций
                eff_p.append(seff)
                # для сбора динамики эффективности от популяции
                seff_lst.append(np.log10(seff[0]))
                ########################################################################

                clear_output()
                ########################################################################
                # сбор|вывод данных эффективности поиска от популяции к популяции
                try:
                    avlmdl = round(avl_mdl/(avl_mdl+non_mdl),2)
                    avlmdl_lst.append(avlmdl)
                    nonmdl = round(non_mdl/(avl_mdl+non_mdl),2)
                    gdmdl = round(gd_mdl/(avl_mdl),2)
                    gdmdl_lst.append(gdmdl)
                    ntkmdl = round(ntk_mdl/(avl_mdl),2)
                    # Показываем ход генитеческого поиска
                    show_pocess(avlmdl_lst, gdmdl_lst, sval_lst, seff_lst)
                    print()
                    print(f'Мегапопуляция {m}, эпоха {it+1}')
                    print(f'Модели популяции: создано {[avl_mdl,avlmdl]} брак {[non_mdl,nonmdl]}')
                    print(f'Модели популяции: пригодны {[gd_mdl,gdmdl]} не пригодны {[ntk_mdl,ntkmdl]}')
                    print()
                except:
                    pass

            ############################################################################
            # сбор и вывод данных для оценки эффективности поиска от эпохи
            # сорт. список точностей мегапопуляции отсортированые точностями популяций
            sval_p = sorted(val_p, key=lambda x: x[0])
            # сортируем список точностей мегапопуляции по эффективности
            seff_p = sorted(eff_p, key=lambda x: x[0])
            eph_time = time.time() - curr_time
            # собираем для контроля поиска и отображения
            svalp_lst.append(np.log10(sval_p[0][0]))
            seffp_lst.append(np.log10(seff_p[0][0]))
            ephtime_lst.append(eph_time)

            ############################################################################
            # Показываем ход генитеческого поиска
            if it > 1: show_process(svalp_lst, seffp_lst, ephtime_lst)

            # Показыааем результаты лучших ботов
            acc_models = np.array(sval_p)[:4,:4]
            eff_models = np.array(seff_p)[:4,:4]
            to_end = round(np.mean(ephtime_lst)*(epohs-it-1))
            print(f'Эпоха {it}, точность моделей {acc_models}')
            print(f'Время на эпоху {eph_time}, эффективность моделей {eff_models}')
            print(f'До окончания поиска {to_end} сек.')
            print(f'К-ко ошибок в длине ботов {err_gen}')
            print()
            ############################################################################

            ########## пересохраняем каждую эпоху данные ###############################
            saver([mega_popul, botpop_lst, mega_info, svalp_lst, ephtime_lst], directory, globals_dict)
            # это можно взять в свернутый лист автопосев на случай сбоя колаба
            # то возобновить код с момента создания новых мега популяций и популяций
            ############################################################################

            ############################################################################
            #   ФОРМИРОВАНИЕ НОВЫХ ПОПУЛЯЦИЙ НА ОСНОВЕ ПОЛУЧЕННЫХ ОЦЕНОК               #
            ############################################################################
            newmega_popul = []
            newmega_info  = []
            newbotpop_lst = []
            # записываем данные лучших по точночти сетей
            # проходимся по выжившим мега-популяциям
            for m in range(psurv):
                # индекс sval из списка сортированных точностей
                idx_p = val_p.index(sval_p[m])
                # получаем оценка популяции idx_p
                val = sorted(raw_val_p[idx_p], reverse=0)

                blockov_list = mega_info[idx_p] # берем информацию
                bot_pop = botpop_lst[idx_p]     # берем мегабота популяцию
                popul = []
                # записываем отобранных ботов в популяции
                for i in range(nsurv):
                    # индекс бота лучших в первичном списке val_p
                    bot_id = val_p[idx_p].index(val[i])

                    # Берем очередного бота
                    bot = mega_popul[idx_p][bot_id]
                    # пополяем популяцию ботом
                    popul.append(bot)
                # пополяем мегапопуляцию
                newmega_popul.append(popul)
                newmega_info.append(blockov_list)
                newbotpop_lst.append(bot_pop)


            # записываем эффективных  ботов популяций
            for m in range(psurv_eff):
                # индекс из списка эффективных
                idx_p = eff_p.index(seff_p[m])
                # эффективность  популяция idx_p
                seff = sorted(raw_eff_p[idx_p], reverse=0)

                blockov_list = mega_info[idx_p] # берем информацию
                bot_pop = botpop_lst[idx_p]     # берем мегабота популяцию
                popul = []
                # записываем отобранных ботов в популяции
                for i in range(nsurv_eff):
                    # индекс бота эффекивных в первичном списке eff_p
                    bot_id = eff_p[idx_p].index(seff[i])

                    # Берем очередного бота
                    bot = mega_popul[idx_p][bot_id]
                    # пополяем популяцию ботом
                    popul.append(bot)
                # пополяем мегапопуляцию
                newmega_popul.append(popul)
                newmega_info.append(blockov_list)
                newbotpop_lst.append(bot_pop)


            # идем по отобранным популяциям точных и эффективных
            for m in range(psurv+psurv_eff):
                #newpopul = []
                popul = newmega_popul[m]
                blockov_list = newmega_info[m]
                bot_pop = newbotpop_lst[m]
                add_n = n - len(newmega_popul[m])
                print(f'Мегапопуляция {m} добавляем {add_n} ботов')

                # берем ген определяющий тип сети популяции
                type_net = bot_pop[2]
                choosing_net = Set_net(type_net, activ_lays, activ_out, neiro_out)

                # инициализируем класс структуры блоков на основе параметров сети
                maker_blocks = Make_blocks(choosing_net)
                bots_poprandom = [maker_blocks.buildblock_bot(blockov_list) \
                                for i in range(add_n)]

                # Проходимся в цикле add_n-раз
                real_parents = min(len(popul), parents_n)
                print(f'Популяция {m}, ботов {len(newmega_popul[m])}')
                # Проходимся в цикле add_n-раз
                for i in range(add_n):
                    idxp =  np.random.randint(0, len(popul), real_parents)
                    if len(popul) == 1: bots_parent = [popul[0]]
                    else: bots_parent = [popul[b] for b in idxp]

                    newbot = []  # Создаем пустой список под значения нового бота
                    # Пробегаем по всей длине бота
                    for j in range(len(bots_parent[0])):
                        if len(bots_parent[0]) == 1: k=0
                        else: k = np.random.randint(0, real_parents)
                        # есои боты разной
                        try:
                            x = bots_parent[k][j]
                        except:
                            x = bots_poprandom[k][j]
                            err_gen+=1
                            print(f'Бот другой длины, берем случайный ген, к-ко ошибок {err_gen}')
                        # С вероятностью mutn устанавливаем значение бота
                        if (np.random.random() < mutn):
                            k = np.random.randint(0, add_n)
                            x = bots_poprandom[k][j]

                        newbot.append(x)      # Доб. очередное значение в нового бота
                    newmega_popul[m].append(newbot) # Доб. бота в новую популяцию
                    # Для контроля кода - не разрастается ли популяция из-за ошибки
                    print(f'Популяция {m}, дополнена до ботов {len(newmega_popul[m])}')
                # Для контроля кода - не разрастается ли популяция из-за ошибки
                print(f'Популяция {m}, ботов {len(newmega_popul[m])}')

                ########## пересохраняем каждую эпоху данные ###########################
                saver([newmega_popul, newbotpop_lst, newmega_info], directory, globals_dict)
                # это можно взять в свернутый лист автопосев на случай сбоя колаба
                # то возобновить код с момента создания новых мега популяций и популяций
                ########################################################################

            ##################### ПОДСКАЗКА ДЛЯ УЛУЧШЕНИЯ ##############################
            # Можно тут организовать тригер, чтобы, имея сохраненными newmega_popul,
            # newbotpop_lst, newmega_info на диске, запускать цихл не вначале, а с этого
            # момента, загрузив вначале их в код.
            ############################################################################

            ########### Сортировка и сохранение лушх сетей #############################
            # определяем к-во возможных сетей real_bestnets при малых к-вах
            real_bestnets = min(len(newmega_popul[0]), waitnets)
            # получаем индексы  лучших сетей по всем мегапопуляциям
            idxs, sval_best = get_idxbest(sval_p, real_bestnets)
            # получаем спиок лучших сетей по всем мегапопуляциям на эпохе
            newbestnets = get_bestnets(idxs, newbotpop_lst, newmega_info, newmega_popul)
            # получаем список лучших сетей
            for i in range(real_bestnets): bestnets.append(newbestnets[i])
            # объединяем списки
            val_best = np.hstack((val_best,sval_best))
            # получаем индексы для сортировки
            idx = np.argsort(val_best)[:real_bestnets]
            val_best = val_best[idx]
            bestnets = [bestnets[i] for i in idx]
            # сохраняем список лучших сетей
            np.save(directory + 'bestnets.npy',  np.array(bestnets, dtype=object))
            ############################################################################

            ############# ПОДСЕВ лучшх сетей с прошлых эпох ############################
            if (it > 1 and  it % frbest == 0) and (p - psurv - psurv_eff - sevbest) > 0:
                pbest = sevbest
                posev_lst = bestnets[:pbest]
                # сеем популяции из листа posev
                for m in range(pbest):
                    bot_pop = posev_lst[m][0]
                    blockov_list = posev_lst[m][1]
                    popul = []  # Создаем пустую популяцию
                    # задаем первого бота популяции из сева
                    popul.append(posev_lst[m][2])
                    # Проходим по всей длине популяции от 1
                    for i in range(1, nnew):
                        # берем ген определяющий тип сети популяции
                        type_net = bot_pop[2]
                        choosing_net = Set_net(type_net, activ_lays, activ_out,
                                            neiro_out)
                        # иници/ класс структуры блоков на основе парраметров сети
                        maker_blocks = Make_blocks(choosing_net)
                        # создаем оставшихся случайнонных ботов из сеяного blockov_list
                        bot = maker_blocks.buildblock_bot(blockov_list)
                        popul.append(bot)      # добавляем бота в популяцию
                    # добавляем популяцию в мегапопуляцию
                    newmega_popul.append(popul)
                    # добавляем информацию о блоках  в мегапопуляцию
                    newmega_info.append(blockov_list)
                    # добавляем мегабота популяции в лист ботов популяций
                    newbotpop_lst.append(bot_pop)
                print(f'Посеено {len(posev_lst)} сетей из лучших прошлых эпох')
            else:
                pbest = 0

            ############################################################################
            #  досоздаем ботов к отобранным популяциям основе скрещиваний и мутаций   #
            ############################################################################
            # сколько  создаем новых популяций от родителей
            pneweph = max(0, pnew - pbest)  # Кол-во новых популяций
            if pneweph:
                # случайно выбираем тип сети
                type_net =  random.choice(list(style_net.keys()))
                # создаем новых ботов в эту популяцию
                choosing_net = Set_net(type_net, activ_lays, activ_out, neiro_out)
                # инициализируем класс структуры блоков на основе парраметров сети
                maker_blocks = Make_blocks(choosing_net)
                bots_poprandom = [maker_blocks.buildpopulbot(q_tyblocks, q_lays) \
                                for i in range(pnew - pbest)]
                # контроль к-ва родителей при малых популяциях
                real_parents_p = min(len(newbotpop_lst), parents_p)
                # создаем популяции от родителей
                for m in range(pneweph):
                    idxp =  np.random.randint(0, psurv + psurv_eff - 1, real_parents_p)
                    # контроль для малого к-ва мегапопуляций
                    if len(newbotpop_lst) == 1: bots_popparent = newbotpop_lst[0]
                    else: bots_popparent = [newbotpop_lst[i] for i in idxp]

                    # Создаем пустой список под значения нового мегабота
                    newbot_pop = []
                    # Получаем случайное число в диапазоне от 0 до 1
                    if np.random.random() < prb_randbot:
                        k = np.random.randint(0, pneweph)
                        # берем совсем случайный бот пупуляции
                        bot_pop = bots_poprandom[k]

                    else:  # создаем бот пупуляции генетикой
                        # Пробегаем по всей длине бота
                        for j in range(len(bots_popparent[0])):
                            if len(bots_popparent[0]) == 1: k=0
                            else: k = np.random.randint(0, real_parents_p-1)
                            x = bots_popparent[k][j]

                            # С вер. mutp ставим значение бота из случайного мегабота
                            # только не трогаем тип сети мутацией
                            if np.random.random() < mutp and j!=2:
                                k = np.random.randint(0, pneweph)
                                x = bots_poprandom[k][j]
                            newbot_pop.append(x)  # Доб. ген в нового мегабота
                        bot_pop = newbot_pop      # бот популяции создан

                    # генерируем из состав блоков популяции
                    structure  = [np.random.randint(0, bot_pop[1]) for i in \
                                range(bot_pop[0])]
                    # создаем единый список блоков для популяции
                    blockov_list = maker_blocks.sostav_blockov(structure)

                    popul = [] # Создаем пустую популяцию
                    # Проходим по всей длине популяции
                    for i in range(nnew):
                        # создаем очередного случайнонного бота на основе blockov_list
                        bot = maker_blocks.buildblock_bot(blockov_list)
                        popul.append(bot)  # доб. бота в популяцию

                    newmega_popul.append(popul)       # доб. популяцию в мегапопуляцию
                    newmega_info.append(blockov_list) # доб. информацию о блоках
                    newbotpop_lst.append(bot_pop)     # доб. мегабота
            # Для контроля кода - не разрастается ли популяция из-за ошибки
            print(f'Всех популяций {len(newmega_popul)}, ботов {len(newmega_popul[0])}')

            # перезаписываем информацию
            mega_popul = newmega_popul
            mega_info = newmega_info
            botpop_lst = newbotpop_lst
            ############################

            ########## пересохраняем каждую эпоху данные ###############################
            saver([mega_popul, botpop_lst, mega_info, svalp_lst, ephtime_lst], directory, globals_dict)
            # это можно взять в свернутый лист автопосев на случай сбоя колаба
            # и возобновить код с момента создания новых мега популяций и популяций
            ############################################################################
            # удаляем лишнее
            del(newmega_popul)
            del(newmega_info)
            del(newbotpop_lst)
            # чистим память
            gc.collect()

        finish_time = time.time() - start_time
        print(f'Общее время подбора за {epohs} эпох по {p*n} моделей составило {finish_time}')
        print(f'К-ко ошибок в длине ботов {err_gen}')