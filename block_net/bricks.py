import numpy as np # библиотека нампи
import random as random # Импортируем модель randim
import tensorflow as tf         # библиотека машинного обучения
import inspect # для получения имен аргументов функции

# Класс для определения слое
class Set_net():
    '''
    Класс для формирования списка списков блоков,
    имен слоев и значений парамметров слоев
    type_net       - тип сети:
                     0 - Dense
                     1 - Conv
                     2 - Recur
                     None - любая
    activ_lays     - список функций активаций
    activ_out      - выходная функция активации
    neiro_out      -  количество нейронов/сверток выходного слоя
    limit          -  ограничения роста сети
                      по умолчание 10**3
    '''
    def __init__(self,
                 type_net,      # тип сети
                 activ_lays,    # список функций активаций
                 activ_out,     # выходная функция активации
                 neiro_out,     # количество нейронов/сверток выходного слоя
                 limit = 10**3, # ограничения роста сети
                 ) -> None:

        # списки имен используемых слоев
        self.main_lays =  ['Dense', 'Conv1D']
        self.recur_lays = ['Recur', 'EmbRecur']
        self.add_lays =   ['SCnv1D', 'Conv1DT',
                           'Conv1D_dilation_block',
                           'Inceptionv_1D']
        self.optim_lays = ['Dropout','BatchN', 'LayNorm']
        self.pooling_lays = ['MaxP1D','AvgP1D', 'Ups1D']
        self.embed_lays =   ['Embed']
        self.activ_lays = activ_lays
        self.activ_out = activ_out
        self.neiro_out = neiro_out

        # все используемые именя слоев
        self.use_layers  = self.main_lays + self.recur_lays + self.optim_lays\
                           + self.pooling_lays + self.embed_lays\
                           + self.add_lays + ['activ',]

        # создаем self переменные
        self.type_net = type_net
        self.limit = limit
        self.layer = tf.keras.layers

        # создаем список функций слоев
        self.makers_layer = (self.make_dense,
                             self.make_conv1D,
                             self.make_recur,
                             self.make_embrecur,
                             self.make_dropout,
                             self.make_batchn,
                             self.make_laynorm,
                             self.make_maxp1D,
                             self.make_avg1D,
                             self.make_upsam1D,
                             self.make_embedding,
                             self.make_sepconv1D,
                             self.make_conv1DT,
                             self.make_convblock_dilation,
                             self.make_inceptionv_1D,
                             self.make_activ,
        )
        # создаем словарь соответсвия имен и функций слоев
        self.dict_layers = dict(zip(self.use_layers, self.makers_layer))
        pass


    # Функции создания слоев
    def make_dropout(self, x, rate):
        lay = self.layer.Dropout(rate)(x)
        return lay

    def make_batchn(self, x):
        lay = self.layer.BatchNormalization()(x)
        return lay

    def make_laynorm(self, x):
        lay = self.layer.LayerNormalization()(x)
        return lay

    def make_dense(self,x, neiron):
        lay = self.layer.Dense(neiron)(x)
        return lay

    def make_conv1D(self, x, filter, kernel, pads):
        # корректируем размерность под conv1D
        if len(x.shape) < 3:
            # высчитываеи корректировку размерности тензора под lstm
            newshape = self.__redim__(x.shape[1], 2, sort = 0)
            x = tf.keras.layers.Reshape(newshape)(x)

        lay = self.layer.Conv1D(filters = filter,
                                kernel_size = kernel,
                                padding = pads)(x)
        return lay

    def make_sepconv1D(self, x, filter, kernel, pads):
        # корректируем размерность под Separableconv1DT
        if len(x.shape) < 3:
            # высчитываеи корректировку размерности тензора под lstm
            newshape = self.__redim__(x.shape[1], 2, sort = 0)
            x = tf.keras.layers.Reshape(newshape)(x)

        lay = self.layer.SeparableConvolution1D(filters = filter,
                                                kernel_size = kernel,
                                                  padding = pads)(x)
        return lay

    def make_conv1DT(self, x, filter,  kernel, pads):
        # корректируем размерность под conv1DT
        if len(x.shape) < 3:
            # высчитываеи корректировку размерности тензора под lstm
            newshape = self.__redim__(x.shape[1], 2, sort = 0)
            x = tf.keras.layers.Reshape(newshape)(x)

        lay = self.layer.Conv1DTranspose(filters = filter,
                                          kernel_size = kernel,
                                          padding = pads)(x)
        return lay


    def make_maxp1D(self, x, pool):
        lay = self.layer.MaxPooling1D(pool_size = pool)(x)
        return lay

    def make_avg1D(self, x, pool):
        lay = self.layer.AveragePooling1D(pool_size = pool)(x)
        return lay

    def make_upsam1D(self, x, upsize):
        lay = self.layer.UpSampling1D(size = upsize)(x)
        return lay

    def get_recurlay(self, type_recur):
        '''
        определение типа рекурентной сети по параметру
        '''
        if type_recur == 'lstm':
           layer = self.layer.LSTM
        elif type_recur == 'gru':
          layer = self.layer.GRU
        return layer

    def make_recur(self, x, type_recur, recuron, sequences):
        layer = self.get_recurlay(type_recur)
        # корректируем размерность под рекурентный слой
        if len(x.shape) < 3:
            # высчитываеи корректировку размерности тензора под lstm
            newshape = self.__redim__(x.shape[1], 2, sort = 0)
            x = tf.keras.layers.Reshape(newshape)(x)

        lay = layer(units = recuron, return_sequences = sequences)(x)
        return lay

    def make_embedding(self, x, in_emb, out_emb):
        # корректируем размерность под эмбединг
        if len(x.shape) > 2: x = self.layer.Flatten()(x)
        lay = self.layer.Embedding(input_dim = in_emb,
                                             output_dim = out_emb)(x)
        return lay

    def make_embrecur(self, x, recuron, type_recur):
        layer = self.get_recurlay(type_recur)
        # корректируем размерность под рекурентный слой
        if len(x.shape) > 2:  x = self.layer.Flatten()(x)
        # парамметры для Embedding слоя
        emb_in = x.shape[1]
        emb_out = min(64, emb_in//3)
        lay = self.layer.Embedding(input_dim = emb_in,
                                              output_dim = emb_out)(x)
        lay = layer(recuron,return_sequences=True)(lay)
        lay = layer(recuron,return_sequences=False)(lay)
        return lay

    def make_convblock_dilation(self, x, filter, kernel):
        def convs(x, f, k, rate, ln = False):
            x = self.layer.Conv1D(filters = f,
                                  kernel_size = k,
                                  padding = "causal",
                                  dilation_rate = rate,
                                  activation="sigmoid")(x)
            x = self.layer.LayerNormalization()(x) if ln else x
            return x
        # корректируем размерность под conv1DT
        if len(x.shape) < 3:
            # высчитываеи корректировку размерности тензора под lstm
            newshape = self.__redim__(x.shape[1], 2, sort = 1)
            x = tf.keras.layers.Reshape(newshape)(x)

        a = convs(x = x, f = filter,  k = kernel, rate = 2, ln = True)
        b = convs(x = x, f = filter,  k = kernel, rate = 4, ln = True)
        c = convs(x = x, f = filter,  k = kernel, rate = 8, ln = True)
        d = convs(x = x, f = filter,  k = kernel, rate = 16,ln = True)
        lay = self.layer.concatenate([x, a, b, c, d], axis = -1)
        return lay

    def make_inceptionv_1D(self, x, set_filters, kernel_sets, pad_sets):
        '''
        Блок Inseption - как пример сложных блоков для отбора
        Опробирован один тип блока - Inception A
        пример отсюда - https://github.com/Sakib1263/
        Inception-InceptionResNet-SEInception-SEInceptionResNet-1D-2D-Tensorflow-Keras/
        blob/main/Codes/Inception_1DCNN.py
        '''
        def Conv_1D_Block(x, filters, kernel, strides=1, padding="same"):

            # 1D Convolutional Block with BatchNormalization
            x = self.layer.Conv1D(filters,
                                       kernel,
                                       strides=strides,
                                       padding=padding,
                                       kernel_initializer="he_normal")(x)
            x = self.layer.BatchNormalization()(x)
            x = self.layer.Activation('relu')(x)
            return x

        # корректируем размерность под conv1DT
        if len(x.shape) < 3:
            # высчитываеи корректировку размерности тензора под lstm
            newshape = self.__redim__(x.shape[1], 2, sort = 0)
            x = tf.keras.layers.Reshape(newshape)(x)

        # Inception Block
        branch1x1 = Conv_1D_Block(x, set_filters[5],
                                     kernel_sets[5],
                                     padding= pad_sets[5])

        branch3x3_ = Conv_1D_Block(x, set_filters[4],
                                     kernel_sets[4],
                                     padding=pad_sets[4])
        branch3x3 = Conv_1D_Block(branch3x3_, set_filters[3],
                                             kernel_sets[3],
                                             padding=pad_sets[3])

        branch5x5_ = Conv_1D_Block(x, set_filters[2],
                                     kernel_sets[2],
                                     padding=pad_sets[2])
        branch5x5 = Conv_1D_Block(branch5x5_, set_filters[1],
                                             kernel_sets[1],
                                             padding=pad_sets[1])

        branch_pool_ = self.layer.MaxPooling1D(pool_size=3, strides=1,
                                                   padding='same')(x)
        branch_pool = Conv_1D_Block(branch_pool_, set_filters[0],
                                                 kernel_sets[0],
                                                 padding=pad_sets[0])

        lay = self.layer.concatenate([branch1x1, branch3x3,
                                           branch5x5, branch_pool],
                                           axis=-1)
        return lay

    def make_activ(self, x, activ):
        lay = self.layer.Activation(activ)(x)
        return lay


    # Метод построение случайного бота блока на основе bot_list блока
    def __buildbot__(self, bot_list):
        '''
        Метод создает случайным образом в заданном диапазоне
        значения пришедщим парраметрам слоев в списке bot_list
        собирает их в список и выдает этот список

        Инсептион только один тип. Вероятно нужно тут выбор типа,
        а сами сочетания под типы уже в блоке

        '''
        # Для Inception A
        insep_flts_sets = [(64, 96, 128, 16, 32, 32),    # Inception Block 1
                           (128, 128, 192, 32, 96, 64),  # Inception Block 2
                           (192, 96, 208, 16, 48, 64),   # Inception Block 3
                           (160, 112, 224, 24, 64, 64),  # Inception Block 4
                           (128, 128, 256, 24, 64, 64),  # Inception Block 5
                           (112, 144, 288, 32, 64, 64),  # Inception Block 6
                           (256, 160, 320, 32, 128, 128),# Inception Block 7
                           (384, 192, 384, 48, 128, 128) # Inception Block 8

        ]

        insep_krnl_sets = [(1, 1, 3, 1, 3, 1),    # Inception A
                           #(1, 1, 3, 1, 3, 3),   # Inception B
                           #(1, 1, 5, 1, 3, 3),   # Inception C
                           #(1, 1, 7, 1, 7, 7),   # Inception D
        ]

        insep_pads_sets = [('valid','valid','same','valid','same','same'),   # Inception A
                           #('same','same','same','same','same','same'),     # Inception B
                           #('same','same','same','same','same','same'),   # Inception C
                           #(1, 1, 7, 1, 7, 7),   # Inception D
        ]
        bot = []
        for el in bot_list:
            if el == 'neiron':
                bot.append(2**random.randint(2,7))
            if el == 'activ':
                bot.append(random.randint(0, len(self.activ_lays)-1))
            if el == 'filter':
                bot.append(2**random.randint(2,7))
            if el == 'kernel':
                bot.append(random.randint(2,5))
            if el == 'pads':
                bot.append(random.choice(('valid','same')))
            if el == 'stride':
                bot.append(random.randint(1,2))
            if el == 'pool':
                bot.append(random.randint(2,4))
            if el == 'upsize':
                bot.append(random.randint(2,4))
            if el == 'rate':
                bot.append(round(random.random()*0.5,2))
            if el == 'type_recur':
                bot.append(random.choice(('lstm','gru')))
            if el == 'recuron':
                bot.append(random.randint(2,30))
            if el == 'sequences':
                bot.append(random.choice(('False','True')))
            if el == 'in_emb':
                bot.append(random.randint(10,200))
            if el == 'out_emb':
                bot.append(max(20, bot[-1]//3))
            if el == 'set_filters':
                bot.append(random.choice(insep_flts_sets))
            if el == 'kernel_sets':
                bot.append(random.choice(insep_krnl_sets))
            if el == 'pad_sets':
                bot.append(random.choice(insep_pads_sets))
        return bot

    def __redim__(self, size, dim, sort = None):
        '''
        Метод метод считает выходной список размерностей
        длины dim из прешедшего размера парамметров size,
        на основе разложения на простые множители
        input:
        size - входящий размер размерности слоя
        dim  - длина всписка выходной размерности
        sort - сортировать ли список простых множителей
                как - 0 - возраст-й, 1 - убывающий
        output:
        outshape - выходной список размерностей
                   длины dim
        '''
        # собираем в res простые множители чиcла n
        res, n, d = [], size, 2
        while d * d <= n:
            if n % d == 0:
                res.append(d)
                n //= d
            else:
                d += 1
        if n > 1:
            res.append(n)
        ##########################################

        # сортируем или перемешиваем список простых множителей
        if sort == 0 or sort == 1:
            res = sorted(res, reverse=sort)
        else:
            res = np.random.permutation(res)
        ##########################################

        # формируем список размерностей длины dim
        s = len(res)//dim  # целое число отношения len(res) и dim
        if s >= 1 and dim == 2:
            # получим outshape размерности 2 из произведений частей
            outshape = [np.prod(res[:s]), np.prod(res[s:])]

        elif s >= 1 and dim == 3:
            # получим outshape размерности 3 из произведений частей
            outshape = [np.prod(res[:s]), np.prod(res[s:2*s]),
                         np.prod(res[2*s:])]

        elif s < 1 and dim == 3:
            s = len(res)//(dim-1)
            # получим outshape размерности 2 из произведений частей и
            # добавляем ось справа
            outshape = [int(np.prod(res[:s])), int(np.prod(res[s:])), 1]

        elif (s < 1 and dim == 2) or dim == 1:
            # просто добавляем ось справа
            outshape = [size, 1]
        return outshape
    

# Класс генерация блоков
class Make_blocks():
    '''
    Класс отвечающий за генерацию
    блоков сети на основе данных из
    ранее инициализированного класса set_net

    set_net - класс парамметров сети
    '''

    def __init__(self,
                set_net: object,
                ):
        # переназначенм переменные из класса set_net
        self.set_net = set_net
        self.neiro_out = set_net.neiro_out
        self.main_lays = set_net.main_lays
        self.add_lays = set_net.add_lays
        self.recur_lays = set_net.recur_lays
        self.activ_out = set_net.activ_out
        self.limit = set_net.limit
        self.type_net = set_net.type_net

        # определяем слои сети на основы заданного типа сети
        #############################################
        # если тип не задан
        if self.type_net == None:
            self.net_lays = self.main_lays + self.add_lays + self.recur_lays
        # если рекурентный тип сети
        elif self.type_net == 2:
            self.net_lays = self.recur_lays
        # если 0 - Dense или 1 - Conv тип сети
        else:
            # выбираем из main_lays индексом, который равен типу сети
            self.net_lays = [self.main_lays[self.type_net]]
        #############################################
        # отобранные по типу слою и плюс слои оптимизации
        self.__used_lays__ = self.net_lays + self.set_net.optim_lays

        pass

    # ФУНКЦИИ КОРРЕКЦИИ СПИСКОВ БЛОКОВ
    def __correct__(self, block_list: list, name: str,
                   level = 0, insert = False):
        '''
        Метод коррекции пришедщего списка списка
        block_list - пришедщий список
        level - порог выше котого коррекция
        name - имя добавляемого слоя
        insert - если True то вставит до последнего,
              иначе вставит за последним
        '''
        # подбрасываем монетку
        bias = random.random()
        if bias >= level: # если выше порога, то корректируем для вывода
            if insert: # вставляем
               block_list.insert(-1, name)
            else: # добавляем
               block_list.append(name)
        return block_list

    # ФУНКЦИИ ГЕНЕРАЦИИ СПИСКОВ БЛОКОВ
    def __generateblock__(self,
                        max_lays: int,        # мак.количество слоев в блоке
                        prob_mp = 0.27,       # появление пуллинг слоя
                        prob_el = 0.25,       # Embedding до LSTM
                        prob_ac = 0.23        # появление слоя активации
                        ):
        '''
        Внутренний метод для геннерации списка
        из имен слоев
        max_lays - мак. возможное количество слоев в блоке
        '''
        # определяем количество слоев в блоке не более max_lays
        layers = random.randint(0, max_lays)
        block_list = [] # список для сбора имен слоев
        # итерируемся по слоям
        for i in range(layers):
            if not i: # если 0_й слой
              # вставляем случайное имя из слоев сети
              block_list = self.__correct__(block_list,
                                            random.choice(self.net_lays))
            else:
              # если пришло имя из слоя оптимизации
              if block_list[-1] in self.set_net.optim_lays:
                # вставляем случайное имя из слоев сети
                block_list = self.__correct__(block_list,
                                            random.choice(self.net_lays))
              else: # иначе выбираем из слоев сети и оптимизации
                block_list = self.__correct__(block_list,
                                          random.choice(self.__used_lays__))
              # случайное появление пулинга после cвертки
              if block_list[-1]=="Conv1D" and i < layers:
                # случайно по prob_mp добавляем пуллинг слой
                block_list = self.__correct__(block_list,
                                      random.choice(self.set_net.pooling_lays),
                                      level = prob_mp)
              # случайное появление Embeding переd рекуррентным слоем
              if block_list[-1]=="Recur" and i < layers:
                # случайно по prob_el вставляем Embeding слой
                block_list = self.__correct__(block_list,
                                            self.set_net.embed_lays[0],
                                            level = prob_el,
                                            insert = True)
              # случайное появление активации
              backactiv_1 = (block_list[-1] != 'activ' and i < layers)
              backactiv_2 = (block_list[-2] != 'activ' and 1 < i < layers)
              if backactiv_1 or backactiv_2:
                # случайно по prob_ac добавляем 'activ'
                block_list = self.__correct__(block_list, 'activ',
                                            level = prob_ac)
              # если набрали длину
              if len(block_list) == layers:
                  break

        return block_list


    def sostav_blockov(self, q_lst: list):
        '''
        Метод генерирует список из списков блоков
        размера полученнного из q_lst
        q_lst - список длины количества блоков,
                где значения списка определяет
                количество слоев в блоке
        '''
        blockov_list = []
        for i in range(len(q_lst)):
            block = self.__generateblock__(max_lays = q_lst[i])
            blockov_list.append(block)
        return  blockov_list


    def buildblock_bot(self, block_lst: list):
        '''
        Метод собирает список списков парамметров
        слоев каждого блока будущей сети
        block_lst - список списков слоев будущей сети
        '''
        botov_lst = []
        for i in range(len(block_lst)):
            # получаем список имен парамметров слоев блока
            bot_lst = self.__bot_block__(block_lst[i])
            # получаем список самих парамметров слоев блока из имен парамметров
            bot = self.set_net.__buildbot__(bot_lst)
            botov_lst.append(bot)
        return   botov_lst

    ######################################################################

    # ФУНКЦИИ БОТОВ
    # определение состава бота для блока из созданого случайно block_list
    def __bot_block__(self, block_list: list):
        '''
        Метод формирует список списков парамметров слоев в блоках
        на основе списка имен слоев блока из block_list
        block_list - список имен слоев блока
        '''
        bot_list = []
        for lay in block_list:
          if lay == 'activ':
              # если имя слоя 'activ', то просто добавляем 'activ'
              bot_list.append('activ')
          else: # иначе
              # созданный в set_net словарь соответсвия
              # имен слоев и функций их формирующих
              maker_lay = self.set_net.dict_layers[lay]
              # методом param_layer определяем парамметры
              # у функций формирующих слои
              param = self.__param_layer__(maker_lay)
              # добавляем эти парамметры в bot_list
              if len(param): bot_list += param
              else: pass
        return bot_list


    # Построение случайного бота попупуляции
    def buildpopulbot(self, q_tyblocks: int, q_lays: int):
        '''
        Метод случйно генерирует список, который будет
        ботом_попупаляции сетей
        q_tyblocks - максимально возможное количество блоков в сети
        q_lays - максимально возможное количество слоев в блоках в сети
        '''
        # генерируем количество блоков
        qblocks = random.randint(1, q_tyblocks)
        # генерируем количество слоев в блоке
        genlays = random.randint(1, q_lays)

        # собираем бота популяции
        populbot = []
        # добавляем:
        # ген 0 макс. возможное количество блоков сети популяции
        populbot.append(qblocks)
        # ген 1 макс. возможное количество слоев в блоках сети популяции
        populbot.append(genlays)
        # ген 2 типа данной сети
        populbot.append(self.type_net)
        # ген 3 делать или нет пред_выходном слой
        populbot.append(random.randint(0,1))
        # ген 4 ко-ва нейронов/фильтров пред_выходном слое
        populbot.append(2**random.randint(2,7))
        # ген 5 окон если свертки в пред_выходном слое
        populbot.append(random.randint(2,5))
        # ген 6 делать/нет слой нормализации перед посл-й активацией
        populbot.append(random.randint(0,1))
        # ген 7 какую делаем активацию в выходном слое из списка активаций
        populbot.append(random.randint(0, len(self.set_net.activ_out)-1))
        # ген 8 под сложность сети, будет назначается методом сборки сети
        populbot.append(0)
        # ген 9 под ярусность сети, будет назначается методом сборки сети
        populbot.append(0)
        return populbot


    # ФУНКЦИИ ФОРМИРОВАНИЯ БЛОКОВ
    # без степеней - прямые значения нейронов
    def __buildblock__(self, tensor: object,
                       block_list: list, bot: list):
        '''
        Метод строет блок слоев на основе списка
        имен слоев блока и им соответствующих бота,
        являющегося списком парамметров этих слоев
        input:
        tensor     - входящий тензор
        block_list - список имен слоев блока
        bot        - спискок парамметров этих слоев
        output:
        tensor     - исходящий тензор блока
        '''

        # будет добавляться больше или меньше 0 значение
        # если к-во параметров > или < 1
        b=0
        # иттерируеимя по слоям блока
        for i, lay in enumerate(block_list):
          # созданный в set_net словарь соответсвия
          # имен слоев и функций их формирующих
          maker_lay = self.set_net.dict_layers[lay]
          # получаем колчество парамметров функции слоя
          k = len(self.__param_layer__(maker_lay))
          # формируем список аргументов для функции слоя
          if k: # если есть доп.параметры в слое
            # получаем доп.параметры в слоя
            parametrs = [bot[i+b+j] for j in range(k)]
            # собираем аргументы для активации
            if lay=='activ':
              args = [tensor]+[self.set_net.activ_lays[parametrs[0]]]
            # собираем аргументы для других многопарамметных слоев
            else: args = [tensor]+parametrs
          # собираем аргументы если слой без параммeтров
          else: args=[tensor]
          # формируем слой на основе нужных и собранных args
          tensor = maker_lay(*args)
          # обнавляем b
          b+=k-1
        return tensor # выводим выходной тензор блока


    def __buildblockout__(self,
                      indata: object,    # входные данные
                      bot_pop: list,     # бот популяции (может)
                       ):
        '''
        Метод строет блок слоев на основе списка
        имен слоев блока и им соответствующих бота,
        являющегося списком парамметров этих слоев
        input:
        indata     - входящий тензор
        bot_pop    - бот популяции
        output:
        x     - исходящий тензор блока
        '''
        x = indata

        if bot_pop[2] == 0:
            # Добавление предпоследнего полносвязного слоя
            if bot_pop[3]!=0:
                x = self.set_net.make_dense(x, bot_pop[4])
            # Добавление нормализации перед последним полносвязным слоем
            if bot_pop[6]!=0: x = self.set_net.make_batchn(x)

        elif bot_pop[2] == 1:
            # Добавление предпоследнего conv1D слоя
            if bot_pop[3]!=0:
                # высчитываеи корректировку размерности тензора под conv1D
                newshape = self.set_net.__redim__(x.shape[1], 2, sort = 1)
                x = tf.keras.layers.Reshape(newshape)(x)
                x = self.set_net.make_conv1D(x, bot_pop[4], bot_pop[5],
                                             pads = 'same')
            # Добавление нормализации перед последним полносвязным слоем
            if bot_pop[6]!=0: x = self.set_net.make_batchn(x)

        elif bot_pop[2] == 2:
            # Добавление предпоследнего LSTM слоя
            if bot_pop[3]!=0:
                # высчитываеи корректировку размерности тензора под lstm
                newshape = self.set_net.__redim__(x.shape[1], 2, sort = 1)
                x = tf.keras.layers.Reshape(newshape)(x)
                # выбрал 2 парамметра рекурентного выходного слоя жестко
                x = self.set_net.make_recur(x, 'lstm', bot_pop[4], False)
            # Добавление нормализации перед последним полносвязным слоем
            if bot_pop[6]!=0: x = self.set_net.make_batchn(x)

        else: #pass
            # Добавление нормализации перед последним полносвязным слоем
            if bot_pop[6]!=0: x = self.set_net.make_batchn(x)
        return x

    ###########################################################
    ######         вспомогательные методы класса         ######
    ###########################################################
    def __param_layer__(self, method: object):
        '''
        Метод выводит количеству управляющих
        парамметров пришедшей функции method
        на онове inspect.getfullargspec
        method - функция у которой определяютя
                 количество управляющих парамметров
        '''
        return inspect.getfullargspec(method).args[2:]


    # простое соединение произвольного к-ва блоков
    def __flatconcat__(self, set_blocks: list):
        '''
        Метод конкантенации списка тензоров блоков
        через промежуточный перевод в вектор Flatten
        с контролем пространства паремметров и при
        превышении к-ва размерностей или размера
        паремметров, применение GlobalAveragePooling1D
        взамен Flatten
        '''
        out = []

        # иттерируемся по списку тензоров
        for i in range(len(set_blocks)):
            # не является ли уже вектором
            if set_blocks[i].shape != (None, 1):
                # Берем shape тензора
                control_shape = set_blocks[i].get_shape()
                # Если много размерностей или много парамметров у тензора
                if np.prod(control_shape[1:]) > self.limit:
                    if len(control_shape) == 2:
                       #print('control_shape', control_shape)
                       # высчитываеи корректировку размерности тензора под conv1D
                       newshape = self.set_net.__redim__(control_shape[1], 2, sort = 0)
                       #print('newshape', newshape)
                       lay  = tf.keras.layers.Reshape(newshape)(set_blocks[i])
                    elif len(control_shape) > 2:
                       # сортируем размерности по возрастанию, кроме 0
                       newshape = sorted(control_shape[1:], reverse=1)
                       # перемножаем кроме последнего и берем последний
                       newshape = (np.prod(newshape[:-1]), newshape[-1])
                       #print('newshape', newshape)
                       lay  = tf.keras.layers.Reshape(newshape)(set_blocks[i])
                       # Добавляем слой GlobalAveragePooling1D
                       lay = tf.keras.layers.GlobalAveragePooling1D(keepdims=False)(lay)
                    out.append(lay)
                else:
                    if len(control_shape) > 2:
                      # Добавляем слой Flatten
                      lay = tf.keras.layers.Flatten()(set_blocks[i])
                    else:
                      lay = set_blocks[i]
                    #print(f'shape of lay_{i} = {lay.shape}')
                    out.append(lay)
            else: # Иначе выходим
                break
        if len(out) > 1:
          # Конкантенируем тензоры по axis = -1 и выводим
          try: out = tf.keras.layers.concatenate(out, axis = -1)
          # Если попался слой с shape !=2
          except:
            out_ = []
            for lay in out:
              if lay.shape !=2:
                lay = tf.keras.layers.Flatten()(lay)
              out_.append(lay)
            out = tf.keras.layers.concatenate(out_, axis = -1)
        else: out = out[0]
        return  out
