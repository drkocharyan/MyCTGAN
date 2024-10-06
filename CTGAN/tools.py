import pandas as pd
import numpy as np 

import torch
import torch.nn.functional as F

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler,Normalizer
from sklearn.metrics import silhouette_score



class OneHotEncoder:
    def __init__(self):
        """
        Инициализация кодировщика one-hot.
        """
        self.num_classes = None
        self.categories = None
        self.fitted = False

    def fit(self, categories_series):
        """
        Метод для "обучения" кодировщика: определяет уникальные категории и количество классов на основе данных.

        Параметры:
        - categories_series (pd.Series): серию с категориальными данными.
        """
        # Находим уникальные значения категорий и сортируем их
        self.categories = pd.Series(categories_series.unique()).sort_values().tolist()
        self.num_classes = len(self.categories)
        self.fitted = True

    def transform(self, categories_series):
        """
        Преобразует категориальные данные в one-hot вектора.

        Параметры:
        - categories_series (pd.Series): серию с категориальными данными.

        Возвращает:
        - Тензор one-hot векторов.
        """
        if not self.fitted:
            raise ValueError("OneHotEncoder должен быть обучен методом 'fit' перед использованием 'transform'.")

        # Преобразуем категории в индексы
        category_indices = categories_series.apply(lambda x: self.categories.index(x)).values
        category_tensor = torch.tensor(category_indices, dtype=torch.long)

        # Преобразуем индексы в one-hot представление
        return F.one_hot(category_tensor, num_classes=self.num_classes)

    def inverse_transform(self, one_hot_tensor):
        """
        Преобразует one-hot вектора обратно в категориальные данные.

        Параметры:
        - one_hot_tensor (Tensor): тензор one-hot векторов.

        Возвращает:
        - pd.Series с категориальными данными.
        """
        if one_hot_tensor.size(-1) != self.num_classes:
            raise ValueError("Размерность one-hot векторов не соответствует количеству классов.")

        # Используем torch.argmax для восстановления индексов категорий
        category_indices = torch.argmax(one_hot_tensor, dim=-1).tolist()

        # Преобразуем индексы обратно в категории
        return pd.Series([self.categories[idx] for idx in category_indices])
    






class KMeansAutoCluster:
    def __init__(self, max_clusters=10):
        """
        Инициализация кластеризатора K-Means с автоматическим подбором оптимального количества кластеров.

        Параметры:
        - max_clusters (int): максимальное количество кластеров для поиска.
        """
        self.max_clusters = max_clusters
        self.best_k = None
        self.kmeans_model = None
    def __get_k__(self):

      return self.best_k


    def fit(self, data_np):
        """
        Обучение K-Means с подбором оптимального количества кластеров на основе метода силуэта.

        Параметры:
        - data_np (np.array): данные для кластеризации (numpy array).
        """
        best_score = -1
        previous_score = - 1000
        for k in range(2, self.max_clusters):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(data_np)
            labels = kmeans.labels_

            # Рассчитываем силуэтный коэффициент
            score = silhouette_score(data_np, labels)

            # Если текущее значение силуэта лучше предыдущего — сохраняем результат
            if score > best_score:
                best_score = score
                self.best_k = k
                self.kmeans_model = kmeans


        # print(f"Оптимальное количество кластеров: {self.best_k}")

    def predict(self, data_np):
        """
        Предсказание меток кластеров для новых данных на основе обученной модели.

        Параметры:
        - data_np (np.array): данные для предсказания меток кластеров.

        Возвращает:
        - Метки кластеров для каждого примера.
        """
        if self.kmeans_model is None:
            raise ValueError("Модель KMeans не была обучена. Сначала вызовите метод fit().")

        return self.kmeans_model.predict(data_np)
    

class DataPrepare():
  def __init__(self):

    self.cont_trs_storage = {}
    self.disc_trs_storage = {}
    self.columns_cont = []
    self.columns_disc = []


  # надо обработку нанов
  def transform_discrete(self,column: pd.Series, column_name:str):
    ohe = OneHotEncoder()
    ohe.fit(column)
    self.disc_trs_storage[column_name] = ohe
    data = np.array(ohe.transform(column))
    col_names = [column_name + str(i) for i in range(data.shape[1])]

    self.columns_disc.append(column_name)
    return pd.DataFrame(np.array(ohe.transform(column)), columns= col_names), ohe

  def transform_continuos(self,column, column_name:str, normalizer = 'minmax'):
    n_clusters = column.shape[0]
    if normalizer == 'norm':
      scaler = Normalizer()
    else:
      scaler = MinMaxScaler()
    if column.ndim == 1:
      column = column.to_numpy().reshape(-1,1)
    scaler.fit(column)
    norm_data = np.array(scaler.transform(column))
    # print(n_clusters)
    kmeans = KMeansAutoCluster(n_clusters)
    kmeans.fit(norm_data)
    clusters = pd.Series(kmeans.predict(norm_data))

    ohe = OneHotEncoder()
    classes = [i for i in range(int(kmeans.__get_k__()))]
    ohe.fit(pd.Series(classes))
    classes_names = [f"{column_name}_{i}" for i in classes]
    data = pd.DataFrame(np.array(ohe.transform(clusters)), columns = classes_names)
    data = pd.concat([pd.DataFrame(norm_data, columns = [column_name]), data], axis = 1)


    self.cont_trs_storage[column_name] = (scaler, kmeans, ohe)
    self.columns_cont.append(column_name)

    return data


  def transform(self,data: pd.DataFrame, discrete_columns:list):
    columns = data.columns.sort_values()
    new_df = pd.DataFrame()
    for col in columns:
      if col in discrete_columns:
        embedded_col, ohe = self.transform_discrete(data[col], col)

      else:
        embedded_col = self.transform_continuos(data[col],col)


      new_df = pd.concat([new_df, embedded_col], axis = 1)
      cols = new_df.columns[~new_df.columns.isin(self.columns_cont)].sort_values().to_list() + self.columns_cont
    return new_df



class DataPrepare():
  def __init__(self):

    self.cont_trs_storage = {}
    self.disc_trs_storage = {}
    self.columns_cont = []
    self.columns_disc = []


  # надо обработку нанов
  def transform_discrete(self,column: pd.Series, column_name:str):
    ohe = OneHotEncoder()
    ohe.fit(column)
    self.disc_trs_storage[column_name] = ohe
    data = np.array(ohe.transform(column))
    col_names = [column_name + str(i) for i in range(data.shape[1])]

    self.columns_disc.append(column_name)
    return pd.DataFrame(np.array(ohe.transform(column)), columns= col_names), ohe

  def transform_continuos(self,column, column_name:str, normalizer = 'minmax'):
    n_clusters = column.shape[0]
    if normalizer == 'norm':
      scaler = Normalizer()
    else:
      scaler = MinMaxScaler()
    if column.ndim == 1:
      column = column.to_numpy().reshape(-1,1)
    scaler.fit(column)
    norm_data = np.array(scaler.transform(column))
    # print(n_clusters)
    kmeans = KMeansAutoCluster(n_clusters)
    kmeans.fit(norm_data)
    clusters = pd.Series(kmeans.predict(norm_data))

    ohe = OneHotEncoder()
    classes = [i for i in range(int(kmeans.__get_k__()))]
    ohe.fit(pd.Series(classes))
    classes_names = [f"{column_name}_{i}" for i in classes]
    data = pd.DataFrame(np.array(ohe.transform(clusters)), columns = classes_names)
    data = pd.concat([pd.DataFrame(norm_data, columns = [column_name]), data], axis = 1)


    self.cont_trs_storage[column_name] = (scaler, kmeans, ohe)
    self.columns_cont.append(column_name)

    return data


  def transform(self,data: pd.DataFrame, discrete_columns:list):
    columns = data.columns.sort_values()
    new_df = pd.DataFrame()
    for col in columns:
      if col in discrete_columns:
        embedded_col, ohe = self.transform_discrete(data[col], col)

      else:
        embedded_col = self.transform_continuos(data[col],col)


      new_df = pd.concat([new_df, embedded_col], axis = 1)
      cols = new_df.columns[~new_df.columns.isin(self.columns_cont)].sort_values().to_list() + self.columns_cont


    return new_df


class DataSampler():
    # columns/col - признак
# cat - одно из значении дискретного признака


  def __init__(self, data, transformer):
      self.columns = sorted(transformer.columns_disc + transformer.columns_cont)
      self.cols_order = [ ]
      # Сколько всего дискретным признаков (в том числе и те что от cont => n_cont + n_disc)
      self.n_discret = len(transformer.cont_trs_storage) + len(transformer.disc_trs_storage)
      # максимальная размерность ohe из всех (в том числе выжатых из cont)
      self.max_dim_col = max([transformer.disc_trs_storage[i].num_classes for i in transformer.disc_trs_storage ] + [transformer.cont_trs_storage[i][2].num_classes for i in transformer.cont_trs_storage])

      # то в каких строчках (объектах)  встретилась категория j из дискретного признака i
      #  feature_cat_values [i][j]
      self.feature_cat_values = []
      for col in self.columns:
        cat_allocation = [] # в каких строчках встречается cat фичи col (имя признака)

        if col in transformer.columns_disc:
          cat_names = [col + str(i) for i in range(transformer.disc_trs_storage[col].num_classes)]
          

        else:                                     # Взял размерность ohe для этого continuos признака
          cat_names = [col+ '_' + str(i) for i in range(transformer.cont_trs_storage[col][2].num_classes)]

        self.cols_order.append(cat_names)
        #_______________________________________________________________________________
        for cat in cat_names:
          cat_allocation.append(np.nonzero(data[cat])[0])

        self.feature_cat_values.append(cat_allocation)

      self.discrete_col_cond_st = np.zeros(self.n_discret)
      #то сколько категории в i-ой фиче
      self.discrete_col_n_cat =  np.zeros(self.n_discret)

      self._discrete_col_cat_prob = np.zeros((self.n_discret, self.max_dim_col))
      # то сколько всего категории из всех признаков (в том числе выжимки из cont)
      self.n_cat = sum([transformer.disc_trs_storage[i].num_classes for i in transformer.disc_trs_storage ] + [transformer.cont_trs_storage[i][2].num_classes for i in transformer.cont_trs_storage])


        # тут идем чисто по уже дискретным(с выжимкой)
        # col - это массив с именами всех категори типо [name0, name1, name2] 
      current_cond_st = 0

      for col in range(len(self.cols_order)):
        cat_freq = np.sum(data[self.cols_order[col]].to_numpy(), axis = 0)
        
        # print(self.cols_order[col],'\n', cat_freq)

        cat_prob = cat_freq/ np.sum(cat_freq)

        self._discrete_col_cat_prob[col, : cat_prob.shape[0] ]  = cat_prob    
        self.discrete_col_cond_st[col] = current_cond_st
        self.discrete_col_n_cat [col] = len(self.cols_order[col])
        current_cond_st += len(self.cols_order[col])


  def _random_choice_prob_index(self, discrete_col_id):
    # выбирает по индексу 
    probs = self._discrete_col_cat_prob[discrete_col_id]
    r = np.expand_dims(np.random.rand(probs.shape[0]),axis = 1)

    return (probs.cumsum(axis = 1) > r).argmax(axis = 1)

  def _random_choice_prob_name(self, col_names):
    #  выбирает по имени фичи


    _ = pd.DataFrame(self._discrete_col_cat_prob.T, columns = self.columns)
    # id = self.columns.index(col_name)
    probs = _[col_names].to_numpy()

    r = np.expand_dims(np.random.rand(probs.shape[0]),axis = 1)
    
    return (probs.cumsum(axis = 1) > r).argmax(axis = 1)

        
  def sample_condvec(self, batch_dim):
    if self.n_discret == 0:
      return None
    
    # выбрали какие колонки печатать
    discrete_column_id = np.random.choice(np.arange(self.n_discret), batch_dim)

    cond = np.zeros((batch_dim, self.n_cat))
    mask = np.zeros((batch_dim, self.n_discret))
    mask [:, discrete_column_id ] = 1

    cat_id_in_col = self._random_choice_prob_index(discrete_column_id)
    cat_id = self.discrete_col_cond_st[discrete_column_id] + cat_id_in_col
    cond[np.arange(batch_dim), cat_id] = 1

    return cond, mask, discrete_column_id, cat_id_in_col


  def sample_real_condvec(self, batch_dim):
    if self.n_discret == 0:
      return None

    cat_freq = self._discrete_col_cat_prob.flatten()
    cat_freq = cat_freq[cat_freq != 0]
    cat_freq /= np.sum(cat_freq)

    col_ids = np.random.choice(np.arange(len(cat_freq)),batch_dim, p = cat_freq)
    cond = np.zeros((batch_dim, self.n_cat))
    cond [:, col_ids] = 1

    return cond

  def sample_data(self,data, n, col, opt):
    if col is None:
      idx = np.random.randint(len(data), size=n)
      return data[idx]
    
    id = []

    for c,o in zip(col, opt):
      id.append(np.random.choice(self.feature_cat_values[c][o]))
    return data[id]


  def dim_cond_vec(self):
    return self.n_cat
  
  def generate_cond_from_condition_column_info(self, cond_info, batch_dim):
    vec = np.zeros((batch_dim, self.n_cat))
    # id_ = self.
    pass
    # Переписать _____________________________!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!____________________________

