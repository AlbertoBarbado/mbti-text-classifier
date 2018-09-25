# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 11:38:53 2018

@author: alber
"""

# Librerias NLP
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re
import numpy as np
import pandas as pd
import random
import os
from os import listdir
from os.path import isfile, join

import csv
import random
import time
import glob
import pickle
import nltk

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, recall_score, f1_score
from sklearn import preprocessing
from sklearn.utils import shuffle

global stemmer
stemmer = SnowballStemmer("english")

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.models import load_model
from keras.utils.np_utils import to_categorical
from keras.models import load_model

# Semilla aleatoria
# Se usa el RNG por defecto: Mersenne Twister
random.seed(0)
np.random.seed(0)

def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)

def resample(df_data, tipo):
    #median_ref = int(df_data.groupby(tipo).count().median().values[0])
    min_ref = int(df_data.groupby(tipo).count().min().values[0])
    median_ref = min_ref
    df_data = df_data.sort_values(tipo)
    list_cat = list(df_data[tipo].unique())
    
    df_resampled = None
    df_train = None
    df_test = None
    for cat in list_cat:
        df_iter = df_data[df_data[tipo] == cat].copy()
        df_iter = shuffle(df_iter).reset_index()
        del df_iter["index"]
        
        if max(df_iter.index) > median_ref:
            df_iter = df_iter[:median_ref]
        
        if type(df_resampled)==type(None):
            df_resampled = df_iter
            df_train = df_iter[:int(round(0.80*len(df_iter)))]
            df_test = df_iter[int(round(0.80*len(df_iter))):]
        else:
            df_resampled = pd.concat([df_resampled, df_iter])
            df_train = pd.concat([df_train, df_iter[:int(round(0.80*len(df_iter)))]])
            df_test = pd.concat([df_test, df_iter[int(round(0.80*len(df_iter))):]])
            
    # Histograma nuevo
    df_resampled[tipo].hist()
    df_train[tipo].hist()
    df_test[tipo].hist()
    
    print(len(df_train["posts"].max())) # Se ve que los tamaños de textos son mas o menos ctes
    print(len(df_train["posts"].min()))  
        
    return df_resampled, df_train, df_test


def bow_build(df_train):
    #### Construccion de la BOW
    lines = list(df_train["posts"])
    
    # Generar vocabulario
    words = []
    vocabulario = {}
    
    # Primer filtrado de caracteres no utiles
    for line in lines:
        # Elimino links
        line =  re.sub(r"http\S+", "", line)
        # Elimino retweets
        line = re.sub(r"RT", "", line)
        # Elimino otros caracteres especiales
        line = line.replace("[", "")
        line = line.replace("]", "")
        line = line.replace(",", "")
        line = line.replace('"', '')
        line = line.replace('|', ' ')
    
    documento = lines
    
    i = 0
    # Hago la tokenizacion
    for utterance in documento:
        i += 1
        print("iteracion {0}/{1}".format(i, len(documento)))
        # Tokenizo cada frase
        w = re.findall(r'\w+', utterance.lower(),flags = re.UNICODE) # Paso a minusculas todo
        # La añado a la lista
        words.extend(w)
        
    # Uno en una palabra las named_entities
    tokens = words
    _tokens_bef = tokens
    tokens = []
    for chunk in nltk.ne_chunk(nltk.pos_tag(_tokens_bef)):
        if hasattr(chunk, 'label'):
            tokens.append('_'.join([c[0] for c in chunk]))
        else:
            tokens.append(chunk[0])
    words = tokens
    
    
    # Eliminación de las stop_words 
    words = [word for word in words if word not in stopwords.words('english')]
    # Elimino guiones y otros simbolos raros
    words = [word for word in words if not word.isdigit()] # Elimino numeros
    
    # Elimino lo que no sea un valor a-zA-Z
    words = [word for word in words if not hasNumbers(word)]
    words = [re.sub(r'[^a-zA-Z]', "", word) for word in words]
    words = list(filter(None, words)) # Quito las lineas vacias
    
    # Stemming y eliminación de duplicados
    words = [stemmer.stem(w) for w in words]
        
    # Inicializo la bolsa de palabras
    pattern_words = words
    
    # Vocabulario total sin filtrar
    words_totales = sorted(list(set(pattern_words)))
    
#    if len(words)==0:
#        words.append('UNK')
#        words_glob.extend(words)
#        continue            
    
    i = 0
    df_pattern = pd.DataFrame(pattern_words)
    df_pattern['ocurrencias'] = 1
    df_pattern['documento'] = i
    df_pattern.columns = ['palabras', 'ocurrencias', 'documento']
    df_pattern = df_pattern.groupby(['palabras', 'documento'])['ocurrencias'].sum() # En este pundo, al pasarlo a indices, se ordenan
    df_pattern = df_pattern.reset_index()
    
    words_glob = []
    df_pattern_glob = pd.DataFrame()
    
    # Creo Vocabulario
    words =  sorted(list(set(words))) # Ordeno alfabéticamente y elimino duplicados
    words.append('UNK') # Palabra por defecto para las palabras desconocidas
    
    # Pongo las palabras en la lista global de vocabulario
    words_glob.extend(words)
    
    if df_pattern_glob.empty:
        df_pattern_glob = df_pattern.copy()
    else:
        df_pattern_glob = df_pattern_glob.append(df_pattern)
    
    
    # Se eliminan de nuevo las palabras duplicadas
    words_glob =  sorted(list(set(words_glob)))
    
    # Agrego las frecuencias de palabras que se han ido obteniendo
    df_pattern_glob = pd.DataFrame(df_pattern_glob)
    df_pattern_glob = df_pattern_glob.reset_index()
    df_pattern_glob_tot = df_pattern_glob.groupby(["palabras"])["ocurrencias"].sum().copy()
    
    #df_pattern_glob.mean()
    #df_pattern_glob.std()
    #df_pattern_glob.median() # Sale 2, con lo que se ve que casi todas las palabras tienen poca frecuencia
    #df_pattern_glob = df_pattern_glob.sort_values(ascending = False)
    
    # Elimino las palabras que tengan mucha/poca frecuencia
    l_sup = df_pattern_glob_tot.mean() + 2*df_pattern_glob_tot.std()
    df_pattern_glob_tot = df_pattern_glob_tot[df_pattern_glob_tot < l_sup] # Elimino la cola de mas del 95%
    df_pattern_glob_tot = df_pattern_glob_tot[df_pattern_glob_tot > 10] # Elimino las palabras que aparecen menos de 10 veces
    
    #from scipy.stats import normaltest
    #normaltest([0,1,2,2,3,3,3,4,4,5,6])
    
    words_glob = sorted(set(list(pd.DataFrame(df_pattern_glob_tot).
                             reset_index()['palabras']))) # Palabras ya filtradas
    
    words_glob.append('UNK')
    words_tot_glob = df_pattern_glob_tot.to_dict()
    words_tot_glob['UNK'] = 0
    
    return words_glob, df_train, words_totales
    

def crear_vocabulario():

    ## Lectura del documento
    try: 
        #files_path = glob.glob(os.path.abspath('') + '/mbti-myers-briggs-personality-type-dataset/*')
        files_path = glob.glob(os.path.abspath('') + '/mbti-myers-briggs-personality-type-dataset/mbti_1.csv')
    except:
        pass
    
    df_data = None
    
    for f in files_path:
        if type(df_data)==type(None):
                df_data = pd.read_csv(f, engine='python')
        else:
            df_data = pd.concat([df_data, pd.read_csv(f)], engine='python')
            
    
    df_subtypes = df_data["type"].apply(lambda x: pd.Series(list(x)))
    df_subtypes = df_subtypes.rename(columns={0: "energy", 1: "information", 2: "decision", 3: "lifestyle"})
    df_subtypes = df_subtypes.reset_index()
    df_subtypes = df_subtypes.rename(columns={'index':'user_id'})
    
    df_data = df_data.reset_index()
    df_data = df_data.rename(columns={'index':'user_id'})
    
    df_data = df_data.merge(df_subtypes, left_on='user_id', right_on='user_id', how='inner')
    
    
    # Data descriptive analysis
    df_data["type"].hist()
    df_data.groupby("type").count()
    df_data.groupby("type").count().median() # Valor maximo que va a tener de datos cada clase
    df_data.groupby("type").count().mean()
    df_data.groupby("type").count().std()
    
    df_data["energy"].hist()
    df_data.groupby("energy").count()
    df_data.groupby("energy").count().median() # Valor maximo que va a tener de datos cada clase
    df_data.groupby("energy").count().mean()
    df_data.groupby("energy").count().std()
    
    df_data["information"].hist()
    df_data.groupby("information").count()
    df_data.groupby("information").count().median() # Valor maximo que va a tener de datos cada clase
    df_data.groupby("information").count().mean()
    df_data.groupby("information").count().std()
    
    df_data["decision"].hist()
    df_data.groupby("decision").count()
    df_data.groupby("decision").count().median() # Valor maximo que va a tener de datos cada clase
    df_data.groupby("decision").count().mean()
    df_data.groupby("decision").count().std()
    
    df_data["lifestyle"].hist()
    df_data.groupby("lifestyle").count()
    df_data.groupby("lifestyle").count().median() # Valor maximo que va a tener de datos cada clase
    df_data.groupby("lifestyle").count().mean()
    df_data.groupby("lifestyle").count().std()
    
    # Resampling aleatorio de datos hasta ese maximo de la mediana
    """
    Tambien aprovecho para hacer el train/test split
    """
    
    df_resampled_e, df_train_e, df_test_e = resample(df_data, "energy")
    df_resampled_i, df_train_i, df_test_i = resample(df_data, "information")
    df_resampled_d, df_train_d, df_test_d = resample(df_data, "decision")
    df_resampled_l, df_train_l, df_test_l = resample(df_data, "lifestyle")
    
    words_glob_e, df_train_e, words_totales_e = bow_build(df_train_e)
    words_glob_i, df_train_i, words_totales_i = bow_build(df_train_i)
    words_glob_d, df_train_d, words_totales_d = bow_build(df_train_d)
    words_glob_l, df_train_l, words_totales_l = bow_build(df_train_l)
    
    
    vocabulario = [{'corpus':{"energy": words_glob_e, "information": words_glob_i, "decision":words_glob_d, "lifestyle":words_glob_l},
                       'test_files':{"energy": df_test_e, "information": df_test_i, "decision": df_test_d, "lifestyle": df_test_l},
                       'train_files':{"energy": df_train_e, "information": df_train_i, "decision": df_train_d, "lifestyle": df_train_l},
                       'corpus_total':{"energy": words_totales_e, "information": words_totales_i, "decision": words_totales_d, "lifestyle": words_totales_l}}]
        
    # Se guarda el vocabulario global en disco
    with open(os.path.abspath('') + r'/generated_data_2/vocabulario.p', 'wb') as f:
        pickle.dump(vocabulario, f)
    
    return vocabulario


def cargar_corpus():
    with open(os.path.abspath('') + r'/generated_data_2/vocabulario.p', 'rb') as f:
       vocabulario =  pickle.load(f)
    return vocabulario

def cargar_train(tipo):
    with open(os.path.abspath('') + r'/generated_data_2/train_files_'+ tipo +'.p', 'rb') as f:
       train_f =  pickle.load(f)
    return train_f

def cargar_test(tipo):
    with open(os.path.abspath('') + r'/generated_data_2/test_files_'+ tipo +'.p', 'rb') as f:
       test_f =  pickle.load(f)
    return test_f

def cargar_train_rnn(tipo):
    with open(os.path.abspath('') + r'/generated_data_2/train_files_rnn_'+ tipo +'.p', 'rb') as f:
       train_f =  pickle.load(f)
    return train_f

def cargar_test_rnn(tipo):
    with open(os.path.abspath('') + r'/generated_data_2/test_files_rnn_'+ tipo +'.p', 'rb') as f:
       test_f =  pickle.load(f)
    return test_f


def train_generation(vocabulario, tipo):

    train_files = vocabulario[0]['train_files'][tipo]
    corpus = vocabulario[0]['corpus'][tipo]
    X = []
    user_id = list(train_files["user_id"])
    
    ### One-hot encoding labels
    y = list(train_files[tipo])
    labelencoder_X = LabelEncoder()
    onehotencoder = OneHotEncoder()
    
    y = labelencoder_X.fit_transform(y)
    y = y.reshape(-1, 1)
    y = onehotencoder.fit_transform(y).toarray()
    
    # Remove dummy variable trap
    y = y[:, 1:] # Elimino una de las columnas por ser linearmente dependiente de las demas
    
    ### Train text encoding 
    lines = list(train_files["posts"])
        
    # Primer filtrado de caracteres no utiles
    for line in lines:
        # Elimino links
        line =  re.sub(r"http\S+", "", line)
        # Elimino retweets
        line = re.sub(r"RT", "", line)
        # Elimino otros caracteres especiales
        line = line.replace("[", "")
        line = line.replace("]", "")
        line = line.replace(",", "")
        line = line.replace('"', '')
        line = line.replace('|', ' ')
    
    utterances = lines
    
    for text in utterances:
        # Tokenizo cada frase
        w = re.findall(r'\w+', text.lower(),flags = re.UNICODE) # Paso a minusculas todo
        words = w
        
        # Uno en una palabra las named_entities
        tokens = words
        _tokens_bef = tokens
        tokens = []
        for chunk in nltk.ne_chunk(nltk.pos_tag(_tokens_bef)):
            if hasattr(chunk, 'label'):
                tokens.append('_'.join([c[0] for c in chunk]))
            else:
                tokens.append(chunk[0])
        words = tokens
        
        # Eliminación de las stop_words 
        words = [word for word in words if word not in stopwords.words('english')]
        # Elimino guiones y otros simbolos raros
        words = [word for word in words if not word.isdigit()] # Elimino numeros     
        
        # Elimino lo que no sea un valor a-zA-Z
        words = [word for word in words if not hasNumbers(word)]
        words = [re.sub(r'[^a-zA-Z]', "", word) for word in words]
        words = list(filter(None, words)) # Quito las lineas vacias
        
        # Stemming 
        words = [stemmer.stem(w) for w in words]
        # Pongo como UNK las que no estan en el vocabulario - flattening
        words = [x if x in corpus else 'UNK' for x in words]
        
        sentence = [0]*len(corpus)
        
        for t in words:
            if t in corpus:
                idx = corpus.index(t)
                sentence[idx] += 1
        
        # Pongo a 1 las frecuencias de los UNK para no influir demasiado
        sentence[-1] = 1
        
        # Guardo la frase final
        X.append(np.array(sentence)) # lo guardo como un numpy array
    
    train_f = {"X":X, "y":y, "y_original": list(train_files[tipo]), 'user_id': user_id}
    
    # Se guarda el vocabulario global en disco
    with open(os.path.abspath('') + r'/generated_data_2/train_files_'+ tipo +'.p', 'wb') as f:
        pickle.dump(train_f, f)
        
    return X, y


def test_generation(vocabulario, tipo):

    test_files = vocabulario[0]['test_files'][tipo]
    corpus = vocabulario[0]['corpus'][tipo]
    X = []
    user_id = list(test_files["user_id"])
    
    ### One-hot encoding labels
    y = list(test_files[tipo])
    labelencoder_X = LabelEncoder()
    onehotencoder = OneHotEncoder()
    
    y = labelencoder_X.fit_transform(y)
    y = y.reshape(-1, 1)
    y = onehotencoder.fit_transform(y).toarray()
    
    # Remove dummy variable trap
    y = y[:, 1:] # Elimino una de las columnas por ser linearmente dependiente de las demas
    
    ### Train text encoding 
    lines = list(test_files["posts"])
        
    # Primer filtrado de caracteres no utiles
    for line in lines:
        # Elimino links
        line =  re.sub(r"http\S+", "", line)
        # Elimino retweets
        line = re.sub(r"RT", "", line)
        # Elimino otros caracteres especiales
        line = line.replace("[", "")
        line = line.replace("]", "")
        line = line.replace(",", "")
        line = line.replace('"', '')
        line = line.replace('|', ' ')
    
    utterances = lines
    
    for text in utterances:
        # Tokenizo cada frase
        w = re.findall(r'\w+', text.lower(),flags = re.UNICODE) # Paso a minusculas todo
        words = w
        
        # Uno en una palabra las named_entities
        tokens = words
        _tokens_bef = tokens
        tokens = []
        for chunk in nltk.ne_chunk(nltk.pos_tag(_tokens_bef)):
            if hasattr(chunk, 'label'):
                tokens.append('_'.join([c[0] for c in chunk]))
            else:
                tokens.append(chunk[0])
        words = tokens
        
        # Eliminación de las stop_words 
        words = [word for word in words if word not in stopwords.words('english')]
        # Elimino guiones y otros simbolos raros
        words = [word for word in words if not word.isdigit()] # Elimino numeros     
        
        # Elimino lo que no sea un valor a-zA-Z
        words = [word for word in words if not hasNumbers(word)]
        words = [re.sub(r'[^a-zA-Z]', "", word) for word in words]
        words = list(filter(None, words)) # Quito las lineas vacias
        
        # Stemming 
        words = [stemmer.stem(w) for w in words]
        # Pongo como UNK las que no estan en el vocabulario - flattening
        words = [x if x in corpus else 'UNK' for x in words]
        
        sentence = [0]*len(corpus)
        
        for t in words:
            if t in corpus:
                idx = corpus.index(t)
                sentence[idx] += 1
        
        # Pongo a 1 las frecuencias de los UNK para no influir demasiado
        sentence[-1] = 1
        
        # Guardo la frase final
        X.append(np.array(sentence)) # lo guardo como un numpy array
    
    train_f = {"X":X, "y":y, "y_original": list(test_files[tipo]), 'user_id':user_id}
    
    # Se guarda el vocabulario global en disco
    with open(os.path.abspath('') + r'/generated_data_2/test_files_'+ tipo +'.p', 'wb') as f:
        pickle.dump(train_f, f)
        
    return X, y


def train_model(model_type, tipo, X, y):

#    d_train= cargar_train()
#    X = d_train["X"]
#    y = d_train["y"]
#    
    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.1, random_state = 0)

    if model_type == "random_forest":
        # Fitting Random Forest Classificator to the Training set
        classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
        classifier.fit(X_train, y_train)
        print("Entrenamiento terminado")
        
        # Predicting the Test set results
        y_pred = classifier.predict(X_val)
        
        # Deshago el onehot encoding para sacar las metricas
        y_val = pd.DataFrame(y_val)
        y_pred = pd.DataFrame(y_pred)       
        y_val = [(np.argmax(np.asarray(x)) + 1) if max(np.asarray(x)) > 0 else 0.0 for x in y_val.values]
        y_pred = [(np.argmax(np.asarray(x)) + 1) if max(np.asarray(x)) > 0 else 0.0 for x in y_pred.values]

    elif model_type == "naive_bayes":
        ##### Naive-Bayes
        """
        Resultados muy malos
        """
        # Feature Scaling
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_val = sc.transform(X_val)
        
        y_train = [(np.argmax(np.asarray(x)) + 1) if max(np.asarray(x)) > 0 else 0.0 for x in y_train.tolist()]
        
        # Fitting classifier to the Training set
        from sklearn.naive_bayes import GaussianNB
        classifier = GaussianNB() #No tiene argumentos de input
        classifier.fit(X_train, y_train)
        
        y_pred = list(classifier.predict(X_val))
        
        # Making the Confusion Matrix
        y_val = [(np.argmax(np.asarray(x)) + 1) if max(np.asarray(x)) > 0 else 0.0 for x in y_val]
        cm = confusion_matrix(y_val, y_pred)
        # Accuracy
        accuracy = accuracy_score(y_val, y_pred)
        # Precision
        average_precision = precision_score(y_val, y_pred, average = "weighted")
        # Recall
        recall = recall_score(y_val, y_pred, average='weighted')
        # F1
        f1 = f1_score(y_val, y_pred, average='weighted', labels=np.unique(y_pred))
        print("Modelo - resultados")
        print("accuracy ", accuracy, " precision ", average_precision, " recall ", recall, " f1 ", f1)

    elif model_type == "kernel_svm":
        ### Kernel SVM
        """
        kernel rbf = Mejora un poco al Naive
        kernel poly = Peor que rbf
        kernel linear = Mejores resultados que los anteriores!
        kernel sigmoid = Mejores resultados de todos!!!
        """
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_val = sc.transform(X_val)
        
        y_train = [(np.argmax(np.asarray(x)) + 1) if max(np.asarray(x)) > 0 else 0.0 for x in y_train.tolist()]
        
        # Fitting classifier to the Training set
        from sklearn.svm import SVC
        # classifier = SVC(kernel = 'rbf', random_state = 0)
        # classifier = SVC(kernel = 'linear', random_state = 0)
        # classifier = SVC(kernel = 'poly', random_state = 0)
        classifier = SVC(kernel = 'sigmoid', random_state = 0)
        
        classifier.fit(X_train, y_train)
        
        # Predicting the Test set results
        y_pred = list(classifier.predict(X_val))
        y_pred = [int(x) for x in y_pred]
        
        y_val = [(np.argmax(np.asarray(x)) + 1) if max(np.asarray(x)) > 0 else 0.0 for x in y_val.tolist()]
        y_val = [int(x) for x in y_val]
        
        # Making the Confusion Matrix
        cm = confusion_matrix(y_val, y_pred)
        # Accuracy
        accuracy = accuracy_score(y_val, y_pred)
        # Precision
        average_precision = precision_score(y_val, y_pred, average = "weighted")
        # Recall
        recall = recall_score(y_val, y_pred, average='weighted')
        # F1
        f1 = f1_score(y_val, y_pred, average='weighted', labels=np.unique(y_pred))
        print("Modelo - resultados")
        print("accuracy ", accuracy, " precision ", average_precision, " recall ", recall, " f1 ", f1)
        
#        del classifier
#        classifier = SVC(kernel = 'sigmoid', random_state = 0)
#        X = sc.fit_transform(X)
#        y = [(np.argmax(np.asarray(x)) + 1) if max(np.asarray(x)) > 0 else 0.0 for x in y.tolist()]
#        classifier.fit(X, y)

    elif model_type == "knn":
        ### KNN
        """
        Peor que el SVM
        """
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_val = sc.transform(X_val)
        
        y_train = [(np.argmax(np.asarray(x)) + 1) if max(np.asarray(x)) > 0 else 0.0 for x in y_train.tolist()]
        
        # Fitting classifier to the Training set
        from sklearn.neighbors import KNeighborsClassifier
        classifier = KNeighborsClassifier(n_neighbors = 50, metric = 'minkowski', p = 7) #Defino que me interesa la distancia Euclidea
        classifier.fit(X_train, y_train)
        
        # Predicting the Test set results
        y_pred = list(classifier.predict(X_val))
        y_pred = [int(x) for x in y_pred]
        
        y_val = [(np.argmax(np.asarray(x)) + 1) if max(np.asarray(x)) > 0 else 0.0 for x in y_val.tolist()]
        y_val = [int(x) for x in y_val]
        
        # Making the Confusion Matrix
        cm = confusion_matrix(y_val, y_pred)
        # Accuracy
        accuracy = accuracy_score(y_val, y_pred)
        # Precision
        average_precision = precision_score(y_val, y_pred, average = "weighted")
        # Recall
        recall = recall_score(y_val, y_pred, average='weighted')
        # F1
        f1 = f1_score(y_val, y_pred, average='weighted', labels=np.unique(y_pred))
        print("Modelo - resultados")
        print("accuracy ", accuracy, " precision ", average_precision, " recall ", recall, " f1 ", f1)

    elif model_type == "linear_svc":
        ### LinearSVC
        """
        Resultados igual de buenos, mas o menos, que KernelSVM
        """
        from sklearn.svm import LinearSVC
        
        y_train = [(np.argmax(np.asarray(x)) + 1) if max(np.asarray(x)) > 0 else 0.0 for x in y_train.tolist()]
        
        classifier = LinearSVC(random_state=0)
        classifier.fit(X_train, y_train)
        print(classifier.coef_)
        print(classifier.intercept_)
        
        y_pred = classifier.predict(X_val)
        y_pred = [int(x) for x in y_pred]
        
        y_val = [(np.argmax(np.asarray(x)) + 1) if max(np.asarray(x)) > 0 else 0.0 for x in y_val.tolist()]
        y_val = [int(x) for x in y_val]
        
        # Making the Confusion Matrix
        cm = confusion_matrix(y_val, y_pred)
        # Accuracy
        accuracy = accuracy_score(y_val, y_pred)
        # Precision
        average_precision = precision_score(y_val, y_pred, average = "weighted")
        # Recall
        recall = recall_score(y_val, y_pred, average='weighted')
        # F1
        f1 = f1_score(y_val, y_pred, average='weighted', labels=np.unique(y_pred))
        print("Modelo - resultados")
        print("accuracy ", accuracy, " precision ", average_precision, " recall ", recall, " f1 ", f1)

    # Persistencia del modelo entrenado
    with open(os.path.abspath('') + r'/generated_data_2/model_'+ model_type +'_'+ tipo + '_t.p', 'wb') as handle:
            pickle.dump(classifier, handle)
    
    return classifier


def test_model(model_type, tipo, X_test, y_test):
 
    # Carga del modelo entrenado
    with open(os.path.abspath('') + r'/generated_data_2/model_'+ model_type +'_'+ tipo + '_t.p', 'rb') as handle:
            classifier = pickle.load(handle)
    

    if model_type == "random_forest":
        pass
    
    elif model_type == "naive_bayes":
        ##### Naive-Bayes
        """
        Resultados muy malos
        """
        # Feature Scaling
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_test = sc.fit_transform(X_test)
        
        y_test = [(np.argmax(np.asarray(x)) + 1) if max(np.asarray(x)) > 0 else 0.0 for x in y_test.tolist()]
        
        # Predictions
        y_pred = list(classifier.predict(X_test))
        
        # Making the Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        # Accuracy
        accuracy = accuracy_score(y_test, y_pred)
        # Precision
        average_precision = precision_score(y_test, y_pred, average = "weighted")
        # Recall
        recall = recall_score(y_test, y_pred, average='weighted')
        # F1
        f1 = f1_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred))
        print("Modelo - resultados")
        print("accuracy ", accuracy, " precision ", average_precision, " recall ", recall, " f1 ", f1)


    elif model_type == "kernel_svm":
        ### Kernel SVM
        """
        kernel rbf = Mejora un poco al Naive
        kernel poly = Peor que rbf
        kernel linear = Mejores resultados que los anteriores!
        kernel sigmoid = Mejores resultados de todos!!!
        """
        # Feature Scaling
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_test = sc.fit_transform(X_test)
        
        y_test = [(np.argmax(np.asarray(x)) + 1) if max(np.asarray(x)) > 0 else 0.0 for x in y_test.tolist()]
        
        # Predictions
        y_pred = list(classifier.predict(X_test))
        
        # Making the Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        # Accuracy
        accuracy = accuracy_score(y_test, y_pred)
        # Precision
        average_precision = precision_score(y_test, y_pred, average = "weighted")
        # Recall
        recall = recall_score(y_test, y_pred, average='weighted')
        # F1
        f1 = f1_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred))
        print("Modelo - resultados")
        print("accuracy ", accuracy, " precision ", average_precision, " recall ", recall, " f1 ", f1)


    elif model_type == "knn":
        ### KNN
        """
        Peor que el SVM
        """
        # Feature Scaling
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_test = sc.fit_transform(X_test)
        
        y_test = [(np.argmax(np.asarray(x)) + 1) if max(np.asarray(x)) > 0 else 0.0 for x in y_test.tolist()]
        
        # Predictions
        y_pred = list(classifier.predict(X_test))
        
        # Making the Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        # Accuracy
        accuracy = accuracy_score(y_test, y_pred)
        # Precision
        average_precision = precision_score(y_test, y_pred, average = "weighted")
        # Recall
        recall = recall_score(y_test, y_pred, average='weighted')
        # F1
        f1 = f1_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred))
        print("Modelo - resultados")
        print("accuracy ", accuracy, " precision ", average_precision, " recall ", recall, " f1 ", f1)


    elif model_type == "linear_svc":
        ### LinearSVC
        """
        Resultados igual de buenos, mas o menos, que KernelSVM
        """
        # Feature Scaling
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_test = sc.fit_transform(X_test)
        
        y_test = [(np.argmax(np.asarray(x)) + 1) if max(np.asarray(x)) > 0 else 0.0 for x in y_test.tolist()]
        
        # Predictions
        y_pred = list(classifier.predict(X_test))
        
        # Making the Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        # Accuracy
        accuracy = accuracy_score(y_test, y_pred)
        # Precision
        average_precision = precision_score(y_test, y_pred, average = "weighted")
        # Recall
        recall = recall_score(y_test, y_pred, average='weighted')
        # F1
        f1 = f1_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred))
        print("Modelo - resultados")
        print("accuracy ", accuracy, " precision ", average_precision, " recall ", recall, " f1 ", f1)

    
    return cm, accuracy, average_precision, recall, f1


# Cargar corpus y crear sets
vocabulario = cargar_corpus()
train_generation(vocabulario, "energy")
train_generation(vocabulario, "information")
train_generation(vocabulario, "decision")
train_generation(vocabulario, "lifestyle")

test_generation(vocabulario, "energy")
test_generation(vocabulario, "information")
test_generation(vocabulario, "decision")
test_generation(vocabulario, "lifestyle")


#X_e = train_files_e['X']
#for i in X_e:
#    i[-1] = 1
#y_e = train_files_e['y']
#
#X_i = train_files_i['X']
#for i in X_i:
#    i[-1] = 1
#y_i = train_files_i['y']
#
#X_d = train_files_d['X']
#for i in X_d:
#    i[-1] = 1
#y_d = train_files_d['y']
#
#X_l = train_files_l['X']
#for i in X_l:
#    i[-1] = 1
#y_l = train_files_l['y']



### Train
train_files_e = cargar_train("energy")
train_files_i = cargar_train("information")
train_files_d = cargar_train("decision")
train_files_l = cargar_train("lifestyle")

X_e = train_files_e['X']
y_e = train_files_e['y']
user_id_e = train_files_e['user_id']

X_i = train_files_i['X']
y_i = train_files_i['y']
user_id_i = train_files_i['user_id']

X_d = train_files_d['X']
y_d = train_files_d['y']
user_id_d = train_files_d['user_id']

X_l = train_files_l['X']
y_l = train_files_l['y']
user_id_l = train_files_l['user_id']

classifier_e = train_model("kernel_svm", "energy", X_e, y_e)
classifier_i = train_model("kernel_svm", "information", X_i, y_i)
classifier_d = train_model("kernel_svm", "decision", X_d, y_d)
classifier_l = train_model("kernel_svm", "lifestyle", X_l, y_l)


#### Test
test_files_e = cargar_test("energy")
test_files_i = cargar_test("information")
test_files_d = cargar_test("decision")
test_files_l = cargar_test("lifestyle")

X_e = test_files_e['X']
y_e = test_files_e['y']
user_id_e = test_files_e['user_id']

X_i = test_files_i['X']
y_i = test_files_i['y']
user_id_i = test_files_i['user_id']

X_d = test_files_d['X']
y_d = test_files_d['y']
user_id_d = test_files_d['user_id']

X_l = test_files_l['X']
y_l = test_files_l['y']
user_id_l = test_files_l['user_id']


cm, accuracy, average_precision, recall, f1 = test_model("kernel_svm", "energy", X_e, y_e)
cm, accuracy, average_precision, recall, f1 = test_model("kernel_svm", "information", X_i, y_i)
cm, accuracy, average_precision, recall, f1 = test_model("kernel_svm", "decision", X_d, y_d)
cm, accuracy, average_precision, recall, f1 = test_model("kernel_svm", "lifestyle", X_l, y_l)


##### RNN
def create_data_train_rnn(vocabulario, tipo):
    
    # vocabulario = cargar_corpus()
    train_files = vocabulario[0]['train_files'][tipo]
    corpus_total = vocabulario[0]['corpus'][tipo]
    X = []
    user_id = list(train_files["user_id"])
    
#    corpus_total = vocabulario[0]["corpus_total"]
#    train_files = vocabulario[0]['train_files']
    
    word2idx = {'START': 0, 'END': 1} # Start/End Tokens. Inicialmente así mi frase es: START END
    current_idx = 2 # El indice comenzará desde la posición 2
    
    X = []
    Y = []
    
    lines = list(train_files["posts"])
    Y = list(train_files[tipo])
    
    # Primer filtrado de caracteres no utiles
    for line in lines:
        # Elimino links
        line =  re.sub(r"http\S+", "", line)
        # Elimino retweets
        line = re.sub(r"RT", "", line)
        # Elimino otros caracteres especiales
        line = line.replace("[", "")
        line = line.replace("]", "")
        line = line.replace(",", "")
        line = line.replace('"', '')
        line = line.replace('|', ' ')
    
    utterances = lines
    
    i = 1
    for text in utterances:
        
        print("iteracion: {0}/{1}".format(i, len(utterances)))
        
        # Tokenizo cada frase
        w = re.findall(r'\w+', text.lower(),flags = re.UNICODE) # Paso a minusculas todo
        words = w
        
        # Uno en una palabra las named_entities
        tokens = words
        _tokens_bef = tokens
        tokens = []
        for chunk in nltk.ne_chunk(nltk.pos_tag(_tokens_bef)):
            if hasattr(chunk, 'label'):
                tokens.append('_'.join([c[0] for c in chunk]))
            else:
                tokens.append(chunk[0])
        words = tokens
        
        # Eliminación de las stop_words 
        words = [word for word in words if word not in stopwords.words('english')]
        # Elimino guiones y otros simbolos raros
        words = [word for word in words if not word.isdigit()] # Elimino numeros     
        
        # Elimino lo que no sea un valor a-zA-Z
        words = [word for word in words if not hasNumbers(word)]
        words = [re.sub(r'[^a-zA-Z]', "", word) for word in words]
        words = list(filter(None, words)) # Quito las lineas vacias
        
        # Stemming 
        words = [stemmer.stem(w) for w in words]
        
        # Pongo como UNK las palabras que no están en mi vocabulario
        tokens = [x if x in corpus_total else 'UNK' for x in tokens]
        
        sentence = []
            
        for t in tokens:
            if t not in word2idx:
                word2idx[t] = current_idx
                current_idx += 1 # El índice lo voy aumentando a medida que añado tokens
            idx = word2idx[t]
            sentence.append(idx)
        
        # Guardo la frase final
        X.append(np.array(sentence)) # lo guardo como un numpy array
        
        i += 1
        
        
    train_files_rnn = [{'X':X, 'Y':Y, 
                       'current_idx': current_idx, 
                       'word2idx': word2idx, 
                       "user_id": user_id}]
    
    # Persistencia del modelo entrenado
    with open(os.path.abspath('') + r'/generated_data_2/train_files_rnn_'+tipo+'.p', 'wb') as handle:
            pickle.dump(train_files_rnn, handle)
    
    return X, Y, word2idx, current_idx


def create_data_test_rnn(vocabulario, tipo):
    
    # vocabulario = cargar_corpus()
    corpus_total = vocabulario[0]['corpus'][tipo]
    test_files = vocabulario[0]['test_files'][tipo]
    user_id = list(test_files["user_id"])
    
    # Persistencia del modelo entrenado
    with open(os.path.abspath('') + r'/generated_data_2/train_files_rnn_'+tipo+'.p', 'rb') as handle:
            train_files_rnn = pickle.load(handle)
            
    
    word2idx = train_files_rnn[0]["word2idx"]

    X = []
    Y = []
    
    lines = list(test_files["posts"])
    Y = list(test_files[tipo])
    
    # Primer filtrado de caracteres no utiles
    for line in lines:
        # Elimino links
        line =  re.sub(r"http\S+", "", line)
        # Elimino retweets
        line = re.sub(r"RT", "", line)
        # Elimino otros caracteres especiales
        line = line.replace("[", "")
        line = line.replace("]", "")
        line = line.replace(",", "")
        line = line.replace('"', '')
        line = line.replace('|', ' ')
    
    utterances = lines
    
    i = 1
    for text in utterances:
        
        print("iteracion: {0}/{1}".format(i, len(utterances)))
        
        # Tokenizo cada frase
        w = re.findall(r'\w+', text.lower(),flags = re.UNICODE) # Paso a minusculas todo
        words = w
        
        # Uno en una palabra las named_entities
        tokens = words
        _tokens_bef = tokens
        tokens = []
        for chunk in nltk.ne_chunk(nltk.pos_tag(_tokens_bef)):
            if hasattr(chunk, 'label'):
                tokens.append('_'.join([c[0] for c in chunk]))
            else:
                tokens.append(chunk[0])
        words = tokens
        
        # Eliminación de las stop_words 
        words = [word for word in words if word not in stopwords.words('english')]
        # Elimino guiones y otros simbolos raros
        words = [word for word in words if not word.isdigit()] # Elimino numeros     
        
        # Elimino lo que no sea un valor a-zA-Z
        words = [word for word in words if not hasNumbers(word)]
        words = [re.sub(r'[^a-zA-Z]', "", word) for word in words]
        words = list(filter(None, words)) # Quito las lineas vacias
        
        # Stemming 
        words = [stemmer.stem(w) for w in words]
        
        # Pongo como UNK las palabras que no están en mi vocabulario
        tokens = [x if x in corpus_total else 'UNK' for x in tokens]
        
        sentence = []
            
        for t in tokens:
            if t not in word2idx:
                idx = "UNK"
            else:
                idx = word2idx[t]
                sentence.append(idx)
        
        # Guardo la frase final
        X.append(np.array(sentence)) # lo guardo como un numpy array
        
        i += 1
        
        
    test_files_rnn = [{'X':X, 'Y':Y, 'user_id':user_id}]
    
    # Persistencia del modelo entrenado
    with open(os.path.abspath('') + r'/generated_data_2/test_files_rnn_'+tipo+'.p', 'wb') as handle:
            pickle.dump(test_files_rnn, handle)
    
    return X, Y


def crear_modelo_rnn(X, Y_classes, word2idx, V, user_id_e, tipo):
    
    import numpy as np
    
    # Labeling numérico de las clases
    labelencoder_Y = LabelEncoder()
    Y = labelencoder_Y.fit_transform(Y_classes)
    
    #seq_length = int(round(np.median([len(x) for x in X])))
    #seq_length = int(round(min([len(x) for x in X])))
    seq_length = 600

    
#    # Encoding
#    enc = OneHotEncoder()
#    enc.fit(Y.reshape(-1,1))
#    onehotlabels = enc.transform(Y.reshape(-1,1)).toarray()
    
#    labelencoder_X = LabelEncoder()
#    onehotencoder = OneHotEncoder()
#    
#    y = labelencoder_X.fit_transform(y)
#    y = y.reshape(-1, 1)
#    y = onehotencoder.fit_transform(y).toarray()
#    
#    # Remove dummy variable trap
#    y = y[:, 1:] # Elimino una de las columnas por ser linearmente dependiente de las demas
    
    
    M = 50 # tamaño hidden layer
    # V # tamaño del vocabulario
    K = len(set(Y)) # Numero de clases
    
    
    # Hago el padding/truncado de los datos
    max_review_length = seq_length
    X = sequence.pad_sequences(X, maxlen=max_review_length)
    
    # Feature scaling de los datos de entrada
    X = [preprocessing.scale(x) for x in X]
    
    X = np.array(X)
    
    # Defino el dataset de validacion, especifico su tamaño y reservo esa cantidad de datos para ello 
    X, Y = shuffle(X, Y)
    N = len(X)
    Nvalid = round(N/5)
    Xvalid, Yvalid = X[-Nvalid:], Y[-Nvalid:] # Datos que dejo para validad
    X, Y = X[:-Nvalid], Y[:-Nvalid] # Datos que dejo para entrenar
    
    y_train = Y
    y_val = Yvalid
    
    
    top_words = len(word2idx) # palabras del vocabulario
    
    # Pongo los datos de y de forma categórica
    from keras.utils.np_utils import to_categorical
    y_train = to_categorical(y_train, K)
    y_val = to_categorical(y_val, K)
    X_train = X
    X_val = Xvalid
    
    # Remuevo la dummy variable
    #
    
    # Creo el modelo
    """
    Voy a usar word2vec para hacer un embedding del vector de palabras, proyectándolo en ese espacio vectorial 
    de la dimensión que he definido. Este 'embedding layer' va a aprender la posición del vector de cada palabra
    dentro de él.
    - input_dim = top_words # tamaño del vocabulario original
    - output_dim = embedding_vecor_length # tamaño del espacio vectorial donde se hará el embedding del vector de entrada de palabras
    - input_length = max_review_length # tamaño de las secuencias de palabras que se introducirán en el sistema
    
    entrada:
        vector 2D [batch_size, sequence_length]
    salida:
        vector 3D [batch_size, sequence_length, output_dim]
    """
    
    ############ Building the RNN ############
    
    # Creo el modelo
    embedding_vecor_length = 32 
    
    model = Sequential()
    model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length)) # Vector embedding
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(100))
    model.add(Dense(K, activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.fit(X_train, y_train, epochs=300, batch_size=128)
    
    
    ################# Making the prediction ###################

    # Evaluación final del modelo
    scores = model.evaluate(X_val, y_val, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100)) 
    
    # CM
    #p = model.predict_probab(X_test)
    y_pred = list(model.predict_classes(X_val))
    y_test = y_val
    
    y_test = [(np.argmax(np.asarray(x)) ) if max(np.asarray(x)) > 0 else 0.0 for x in y_test]
    #y_test = np.argmax(y_val, axis=1)
    confusion = confusion_matrix(y_test, y_pred)
    
    # Accuracy
#    y_test = y_val
    import numpy as np
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_test, y_pred)
    
    # Precision
    from sklearn.metrics import precision_score
    precision = precision_score(y_test, y_pred, average = "macro")
    
    # Recall
    from sklearn.metrics import recall_score
    recall = recall_score(y_test, y_pred, average='macro') 
    
    # F1
    f1 = f1_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred))
     
    print("Modelo - resultados")
    print("accuracy ", accuracy, " precision ", precision, " recall ", recall, " f1 ", f1)
    
    # Guardo el modelo
    #model.save("rnn_keras_model.h5")
    path_s = os.path.abspath('') + r'/generated_data_2/rnn_keras_model_'+tipo+'.h5'
    model.save(path_s)
    
    return model




X_e, Y_e, word2idx_e, current_idx_e = create_data_train_rnn(vocabulario, "energy")
X_i, Y_i, word2idx_i, current_idx_i = create_data_train_rnn(vocabulario, "information")
X_d, Y_d, word2idx_d, current_idx_d = create_data_train_rnn(vocabulario, "decision")
X_l, Y_l, word2idx_l, current_idx_l = create_data_train_rnn(vocabulario, "lifestyle")

train_files_rnn_e = cargar_train_rnn("energy")
X_e, Y_e, current_idx_e, user_id_e, word2idx_e = (train_files_rnn_e[0]["X"], 
                                                              train_files_rnn_e[0]["Y"],
                                                              train_files_rnn_e[0]["current_idx"],
                                                              train_files_rnn_e[0]["user_id"],
                                                              train_files_rnn_e[0]["word2idx"])
train_files_rnn_i = cargar_train_rnn("information")
X_i, Y_i, current_idx_i, user_id_i, word2idx_i = (train_files_rnn_i[0]["X"], 
                                                              train_files_rnn_i[0]["Y"],
                                                              train_files_rnn_i[0]["current_idx"],
                                                              train_files_rnn_i[0]["user_id"],
                                                              train_files_rnn_i[0]["word2idx"])
train_files_rnn_d = cargar_train_rnn("decision")
X_d, Y_d, current_idx_d, user_id_d, word2idx_d = (train_files_rnn_d[0]["X"], 
                                                              train_files_rnn_d[0]["Y"],
                                                              train_files_rnn_d[0]["current_idx"],
                                                              train_files_rnn_d[0]["user_id"],
                                                              train_files_rnn_d[0]["word2idx"])
train_files_rnn_l = cargar_train_rnn("lifestyle")
X_l, Y_l, current_idx_l, user_id_l, word2idx_l = (train_files_rnn_l[0]["X"], 
                                                              train_files_rnn_l[0]["Y"],
                                                              train_files_rnn_l[0]["current_idx"],
                                                              train_files_rnn_l[0]["user_id"],
                                                              train_files_rnn_l[0]["word2idx"])



model_e = crear_modelo_rnn(X_e, Y_e, word2idx_e, current_idx_e, user_id_e, "energy")
model_i = crear_modelo_rnn(X_i, Y_i, word2idx_i, current_idx_i, user_id_i, "information")
model_d = crear_modelo_rnn(X_d, Y_d, word2idx_d, current_idx_d, user_id_d, "decision")
model_l = crear_modelo_rnn(X_l , Y_l, word2idx_l, current_idx_l, user_id_l, "lifestyle")



#### Test RNN

X_e_t, Y_e_t = create_data_test_rnn(vocabulario, "energy")
X_i_t, Y_i_t = create_data_test_rnn(vocabulario, "information")
X_d_t, Y_d_t = create_data_test_rnn(vocabulario, "decision")
X_l_t, Y_l_t = create_data_test_rnn(vocabulario, "lifestyle")


test_files_rnn_e = cargar_test_rnn("energy")
X_e, Y_e = (test_files_rnn_e[0]["X"], test_files_rnn_e[0]["Y"])

test_files_rnn_i = cargar_test_rnn("information")
X_i, Y_i = (test_files_rnn_i[0]["X"], test_files_rnn_i[0]["Y"])

test_files_rnn_d = cargar_test_rnn("decision")
X_d, Y_d = (test_files_rnn_d[0]["X"], test_files_rnn_d[0]["Y"])

test_files_rnn_l = cargar_test_rnn("lifestyle")
X_l, Y_l = (test_files_rnn_l[0]["X"], test_files_rnn_l[0]["Y"])




def test_modelo_rnn(X, Y_classes, tipo):
        
    path_s = os.path.abspath('') + r'/generated_data_2/rnn_keras_model_'+tipo+'.h5'
    model = load_model(path_s)     
    
        
    # Labeling numérico de las clases
    labelencoder_Y = LabelEncoder()
    Y = labelencoder_Y.fit_transform(Y_classes)
    
    #seq_length = int(round(np.median([len(x) for x in X])))
    #seq_length = int(round(min([len(x) for x in X])))
    seq_length = 600
    
    M = 50 # tamaño hidden layer
    # V # tamaño del vocabulario
    K = len(set(Y)) # Numero de clases
    
    # Hago el padding/truncado de los datos
    max_review_length = seq_length
    X = sequence.pad_sequences(X, maxlen=max_review_length)
    
    # Feature scaling de los datos de entrada
    X = [preprocessing.scale(x) for x in X]
    
    y_test = list(Y)
    
    # Prediciones
    X = np.array(X)
    y_pred = list(model.predict_classes(X))
    
    # CM
    confusion = confusion_matrix(y_test, y_pred)
    
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Precision
    precision = precision_score(y_test, y_pred, average = "weighted")
    
    # Recall
    recall = recall_score(y_test, y_pred, average='weighted')
    
    # F1
    f1 = f1_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred))
    
    print("Resultados")
    print("accuracy ", accuracy, " precision ", precision, " recall ", recall, " f1 ", f1)
    
    return confusion, accuracy, precision, recall, f1


confusion, accuracy, precision, recall, f1 = test_modelo_rnn(X_e, Y_e, "energy")
confusion, accuracy, precision, recall, f1 = test_modelo_rnn(X_i, Y_i, "information")
confusion, accuracy, precision, recall, f1 = test_modelo_rnn(X_d, Y_d, "decision")
confusion, accuracy, precision, recall, f1 = test_modelo_rnn(X_l, Y_l, "lifestyle")