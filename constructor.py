from stopit import threading_timeoutable as timeoutable
import random as random # Импортируем модель randim
import tensorflow as tf           # библиотека машинного обучения
from block_net.constant import MAX_HIDDEN, MESSAGE_1
# Класс сборки нейронки

class WildregressModel():
      '''
      Класс который формирует и выдает сеть

      input_shape - размерность входящих в сеть данных
      control_level_shape - размер допустимого размера парамметров
                            слоев когда требуется применение
                            GlobalAveragePooling1D для понижения
                            размерности
      q_level - количество блоков с которого можно строить
                многоярусную сеть
      '''

      def __init__(self,
                  input_shape: list,
                  control_level_shape = MAX_HIDDEN,
                  q_level = 3
                  ):

          self.input_shape = input_shape
          self.control = control_level_shape
          self.q_level = q_level
          pass

      # Декоратор для контроля времени
      @timeoutable(default = MESSAGE_1)
      def __call__(self,
                  bot_pop: list,
                  bot: list,
                  setblockov: list,
                  blocks: object
                  ):
          '''
          Класс который формирует и выдает сеть на основе
          bot_pop - бот_популяции сетей
          bot - спискок парамметров слоев блоков
          setblockov - списка списков слоев имен блока
          blocks - класс построения блоков
          '''
          # Входной слой
          inputs =  tf.keras.layers.Input(self.input_shape)
          # размерность даннх сети без 0го, которы None
          dim_net = len(self.input_shape) - 1

          ##### отбор блоков с основными слоями для входа нейронки ############
          idx=[] # хранения индексов блоков с основными слями сети
          # отбираем индексы блоков с основными слями сети
          for block in setblockov:
              if [x for x in blocks.net_lays if x in block]:
                idx.append(setblockov.index(block))
          # берем первый по счету, и если был посев, для входа уходит сразу
          in_nb = idx[0]
          # получаем тензор от первого блока
          in_block = blocks.__buildblock__(inputs, setblockov[in_nb],
                                           bot[in_nb])

          #####################################################################
          ############# отбор для скрытых блоков ##############################
          # ищем и оставляем только один пустой блок
          new_setblockov = []
          new_bot = []
          emp = 0 # счетчик пустых блоков

          # Если был посев, то первый блок для входа уходит сразу -
          for i in range(1,len(setblockov)): # не попадет во внутр.блоки
              #print('metka 4')
              if emp == 0 and setblockov[i] == []:
                  new_setblockov.append(setblockov[i])
                  new_bot.append(bot[i])
                  emp+= 1
              elif emp != 0 and setblockov[i] == []:
                  pass
          # оставляем только один пустой блок
          # на его основе создается проброс от входной части до concat
              else:
                  new_setblockov.append(setblockov[i])
                  new_bot.append(bot[i])

          #####################################################################
          #  print('Сборка многоярусной модели')
          #####################################################################
          if len(new_setblockov) > self.q_level:
              ############### БЛОК соединения скрытых блоков  #################
              ########## создание гена для ярусности и  сложности сети ########
              if not bot_pop[9]:
                  # определяем ярусность сети
                  bot_pop[8] = random.choice(np.arange(2,
                                                       len(new_setblockov)-1))
                   # отбор блоков в ярусы
                  bot_pop[9] = [0] + [len(new_setblockov)] \
                                   + sorted(np.random.choice(np.arange(1,
                                            len(new_setblockov)-1),
                                            bot_pop[8], replace=False))
                  tiers = bot_pop[9]  # состав ярусов
              else:
                  tiers = bot_pop[9]  # состав ярусов
            ###################################################################
              brickblock = [] # ссписок для сборв внутренних блоков
              # итеррируемся по ярусам
              for j in range(len(tiers)-1):
                  indata = in_block if not j else concdata
                  hidblock = []
                  # отбираем в conc только возможные блоки
                  for i in range(tiers[j], tiers[j+1]):
                      ##########################################################
                      # создаем внутренний блок
                      hid = blocks.__buildblock__(indata,
                                              new_setblockov[i],
                                              new_bot[i])
                      # собираем список внутренних блоков для конкатенации
                      hidblock.append(hid)
                  # еслм набрали в список внутренних блоков
                  if len(hidblock)>1:
                    # конкатенируем через выпрямления в вектора
                    concdata = blocks.__flatconcat__(hidblock)
                    # ищем замену размерности
                    newshape = blocks.set_net.__redim__(concdata.shape[-1],
                                                dim_net+1,
                                                sort = 0)
                    # трансформируем размерность тензора
                    concdata = tf.keras.layers.Reshape(newshape)(concdata)
                    brickblock.append(concdata)
                  elif len(hidblock)==1: brickblock.append(hidblock[-1])
                  else: pass
              # соединяем блоки
              to_out = blocks.__flatconcat__(brickblock)
              # передаем в метод создания продпоследнего блока
              out_block = blocks.__buildblockout__(to_out, bot_pop)
              # пробрасываем данные с входного блока и соединяем с другими блоками
              out_block = blocks.__flatconcat__([in_block, out_block])

          # Если есть блоки для одноэтажной модели
          elif len(new_setblockov):
          #####################################################################
          #    print('Сборка одноэтажной модели')
          #####################################################################
              hidblock = []
              for i in range(len(new_setblockov)):
                  hid =  blocks.__buildblock__(in_block, new_setblockov[i],
                                           new_bot[i])
                  hidblock.append(hid)
              ################################################################
              # соединяем блоки
              to_out = blocks.__flatconcat__(hidblock)
              # передаем в метод создания продпоследнего блока
              out_block = blocks.__buildblockout__(to_out, bot_pop)
              # пробрасываем данные с входного блока и соединяем с другими блоками
              out_block = blocks.__flatconcat__([in_block, out_block])

          # Если нет блоков, то берем тензор с входного блока
          else:
              # переводим в вектор входной тензор
              in_block_out = blocks.__flatconcat__([in_block])
              # передаем в метод создания продпоследнего блока
              out_block = blocks.__buildblockout__(in_block_out, bot_pop)
              # пробрасываем данные с входного блока и соединяем с другими блоками
              out_block = blocks.__flatconcat__([in_block, out_block])

          # Финальный слой под вашу задачу
          out = tf.keras.layers.Dense(units = blocks.neiro_out,
                                      activation = blocks.activ_out[bot_pop[7]]
                                      )(out_block)
          # формируем граф модели
          model = tf.keras.Model(inputs, out)
          return model