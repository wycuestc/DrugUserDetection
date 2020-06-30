import os
import pandas as pd
import numpy as np
from collections import Counter, OrderedDict
from keras import backend as K
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from keras.models import Model, Input
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding, Bidirectional, concatenate
from keras.optimizers import Adamax
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.activations import relu
from keras.preprocessing.sequence import pad_sequences
from numpy import array, asarray
from numpy import zeros
from AttentionWeightedAverage import AttentionWeightedAverage
from Attention import Attention
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from matplotlib.ticker import MultipleLocator
from sklearn.model_selection import StratifiedKFold
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def mergeDict(impWordsDict_list):
    c = Counter()
    for d in impWordsDict_list:
        c.update(d)
    dict_c = dict(c)
    ordered_dict = OrderedDict(sorted(dict_c.items(), key=lambda x: x[1], reverse=True))
    return ordered_dict

def countNumMostImpWord(texts_l, index_l): # count the number of most important words
    show = []
    for i in range(0, len(index_l)):
        temp_l = []
        for j in range(0, len(index_l[i])):
            temp_l.append(texts_l[i][index_l[i][j]])
        show.append(temp_l)
    result = {}
    for i in range(0, len(index_l)):
        for j in range(0, len(index_l[i])):
            word = texts_l[i][index_l[i][j]]
            result[word] = result.get(word, 0) + 1
    result = {k: result[k] for k in sorted(result, key=result.get, reverse=True)}
    return result

def sequence_to_text(sentences, reverse_word_map):
    # Looking up words in dictionary
    result = []
    for sentence in sentences:
        result.append([reverse_word_map.get(letter) for letter in sentence])
    return result

def buildInput(max_length, inputFile):
    df = pd.read_csv(inputFile, delimiter=',', encoding='latin-1')

    #preprocess the data
    stop = stopwords.words('english')
    X = df.iloc[:,5].apply(lambda x: [item for item in x.split() if item not in stop])
    lemma = WordNetLemmatizer()
    X = X.apply(lambda x: [lemma.lemmatize(item) for item in x])

    #encode and pad sentences
    Y = df.iloc[:,2]
    le = LabelEncoder()
    Y = le.fit_transform(Y)
    X = X.values.tolist() #convert pandas framwork to the list
    t = Tokenizer()
    t.fit_on_texts(X)
    encoded_docs = t.texts_to_sequences(X)
    padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

    return padded_docs, Y, t


def getEmbeddingMatrix(t, embeddingDim): #laad GLoVe embedding and establish word embedding matrix
    vocab_size = len(t.word_index) + 1
    filedir = "/home/yuchen/PycharmProjects/PRAW/venv/src/embedding_glove/glove.6B/glove.6B." + str(embeddingDim) + "d.txt"
    embeddings_index = dict()
    #f = open('/home/yuchen/PycharmProjects/PRAW/venv/src/embedding_glove/glove.6B/glove.6B.100d.txt')
    f = open(filedir)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Loaded %s word vectors.' % len(embeddings_index))

    embedding_matrix = zeros((vocab_size, embeddingDim))
    for word, i in t.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def getResult(max_length, embeddingDim, t, X_train, X_test, Y_train, Y_test, embedding_matrix, method, flagExistEmbedding=False, flagTrainable=False):
    # X_train (713, 100), X_test (306, 100), Y_train (713,), Y_test (306,)  2d numpy.ndarray. Element is number.
    reverse_word_map = dict(map(reversed, t.word_index.items()))

    vocab_size = len(t.word_index) + 1
    inputs = Input(name='inputs', shape=[max_length])

    #define how to use embedding matrix
    if (flagExistEmbedding == False):
        layer = Embedding(vocab_size, embeddingDim, input_length=max_length)(inputs)
    elif (flagTrainable == False):
        layer = Embedding(vocab_size, embeddingDim, weights = [embedding_matrix], input_length=max_length, trainable=False)(inputs)
    else:
        layer = Embedding(vocab_size, embeddingDim, weights = [embedding_matrix], input_length=max_length, trainable=True)(inputs)

    #define the method we use
    if (method == "biLSTM"):
        layer = Bidirectional(LSTM(64))(layer)
    elif (method == "biLSTMAttentionNew"):
        layer1 = Bidirectional(LSTM(64, return_sequences=True))(layer)
        layer2 = Dense(128, activation=relu)(layer)
        #try different attention layer
        #layer2 =  SimpleRNN(128, activation='relu', return_sequences=True)(layer)
        #layer2 = GRU(128, activation='relu', return_sequences=True)(layer)
        #layer2 = LSTM(128, activation='relu', return_sequences=True)(layer)

        [layer, alpha] = Attention(return_attention = True)([layer1, layer2])

    layer = Dense(512, name='FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.2)(layer)
    layer = Dense(1, name='out_layer')(layer)
    layer = Activation('sigmoid')(layer)
    model = Model(inputs=inputs, outputs=layer)

    # plot the figure of loss with preset learning rate
    model.compile(loss='binary_crossentropy', optimizer='Adamax', metrics=['binary_accuracy'])
    history = model.fit(X_train, Y_train, epochs=4, validation_split=0.3)
    # plt.plot(history.history['loss'], label = 'train')
    # plt.plot(history.history['val_loss'], label='test')
    # plt.gca().xaxis.set_major_locator(MultipleLocator(1))
    # plt.title('Figure of Loss on Train and Test Datasets While Training Process (learning rate: 0.001)')
    # plt.xlabel('number of epochs')
    # plt.ylabel('loss')
    # plt.legend()
    # plt.show()

    Y_pred_prob = model.predict(X_test)

    #choose the most important words in each sentence
    if (method == "biLSTMAttentionNew"):
        att_model = Model(model.input, alpha)
        att_model.compile(loss='binary_crossentropy', optimizer='Adamax', metrics=['binary_accuracy'])
        att_pred = att_model.predict(X_test) #attention matrix (306 * 100)

        numTopKeyword = 5 #the number of key words we like to choose
        index = np.argpartition(att_pred, -numTopKeyword)[:, -numTopKeyword:] #select last(largest) numTopKeyword columns in the attention matrix
        rows = np.transpose([np.arange(att_pred.shape[0])])
        index_sorted = index[rows, np.argsort(att_pred[rows, index])] #top k largest index of each row (sorted from small to large)
                                                                    #numpy array #[[9 41 36... 32 40] [ ] ,,, [ ]]
        index_sorted_list = index_sorted.tolist() # list
        texts_list = sequence_to_text(X_test, reverse_word_map)  # 2d list
        impWords_dict = countNumMostImpWord(texts_list, index_sorted_list)
    else:
        impWords_dict = {}

    Y_pred_prob[Y_pred_prob > 0.5] = 1
    Y_pred_prob[Y_pred_prob <= 0.5] = 0
    Y_pred = Y_pred_prob.T.flatten()

    confusionMatrix = confusion_matrix(Y_test, Y_pred)
    tn, fp, fn, tp = confusionMatrix.ravel()

    return tn, fp, fn, tp, impWords_dict


def testOneFold(max_length, embeddingDim, t, padded_docs, Y, embedding_matrix, method, flagExistEmbedding = False, flagTrainable = False):
    num_splits = 5 #5 folds cross-validation
    kfold = StratifiedKFold(n_splits=num_splits, shuffle=False, random_state=1)
    tnSum = 0
    fpSum = 0
    fnSum = 0
    tpSum = 0
    numIter = 3 #number of iteration
    impWordsDict_list = []

    for i in range(numIter):
        for train, test in kfold.split(padded_docs, Y):
            tn, fp, fn, tp, impWords_dict = getResult(max_length, embeddingDim, t, padded_docs[train], padded_docs[test], Y[train], Y[test], embedding_matrix,
                                                    method, flagExistEmbedding, flagTrainable)
            tnSum += tn
            fpSum += fp
            fnSum += fn
            tpSum += tp
            impWordsDict_list.append(impWords_dict)
    acc = (tpSum+tnSum)/(tnSum+fpSum+fnSum+tpSum)
    precision = tpSum/(tpSum+fpSum)
    recall = tpSum/(tpSum+fnSum)
    f1 = 2*precision*recall/(precision+recall)
    impWordsDict_sum = mergeDict(impWordsDict_list)

    print("-------------------------------------------------------------")
    print("The most important words after 5-folds crossvalidation is:\n")
    print(impWordsDict_sum)
    print('TN: {:d}, FP : {:d},  FN: {:d},  TP: {:d}'.format(tnSum, fpSum, fnSum, tpSum))
    return acc, precision, recall, f1

def test(max_length, embeddingDim, t, padded_docs, Y, embedding_matrix, flag):
    lossList = []
    accList =[]

    if flag == "variant1":
        acc, prec, recall, f1 = testOneFold(max_length, embeddingDim, t, padded_docs, Y, embedding_matrix, "biLSTM", False, False)
        print('Test set\n  Accuracy: {:0.3f}\n  Precision: {:0.3f}\n Recall: {:0.3f}\n f1_score: {:0.3f}'.format(acc, prec, recall, f1))
    elif flag == "variant2":
        acc, prec, recall, f1 = testOneFold(max_length, embeddingDim, t, padded_docs, Y, embedding_matrix, "biLSTM", True, False)
        print('Test set\n  Accuracy: {:0.3f}\n  Precision: {:0.3f}\n Recall: {:0.3f}\n f1_score: {:0.3f}'.format(acc, prec, recall, f1))
    elif flag == "variant3":
        acc, prec, recall, f1 = testOneFold(max_length, embeddingDim, t, padded_docs, Y, embedding_matrix, "biLSTM", True, True)
        print('Test set\n  Accuracy: {:0.3f}\n  Precision: {:0.3f}\n Recall: {:0.3f}\n f1_score: {:0.3f}'.format(acc, prec, recall, f1))
    elif flag == "variant4":
        acc, prec, recall, f1 = testOneFold(max_length, embeddingDim, t, padded_docs, Y, embedding_matrix, "biLSTMAttentionNew", False, False)
        print('Test set\n  Accuracy: {:0.3f}\n  Precision: {:0.3f}\n Recall: {:0.3f}\n f1_score: {:0.3f}'.format(acc, prec, recall, f1))
    elif flag == "variant5":
        acc, prec, recall, f1 = testOneFold(max_length, embeddingDim, t, padded_docs, Y, embedding_matrix, "biLSTMAttentionNew", True, False)
        print('Test set\n  Accuracy: {:0.3f}\n  Precision: {:0.3f}\n Recall: {:0.3f}\n f1_score: {:0.3f}'.format(acc, prec, recall, f1))
    elif flag == "variant6":
        acc, prec, recall, f1 = testOneFold(max_length, embeddingDim, t, padded_docs, Y, embedding_matrix, "biLSTMAttentionNew", True, True)
        print('Test set\n  Accuracy: {:0.3f}\n  Precision: {:0.3f}\n Recall: {:0.3f}\n f1_score: {:0.3f}'.format(acc, prec, recall, f1))

    return lossList, accList



def main():
    scriptDir = os.path.dirname(__file__)
    inputFile = os.path.join(scriptDir, "dataCollection/dataset1000/bidataOpiumStreet1020.csv")

    max_length = 100 #the max length of the sentence
    embeddingDim = 100 #the dimension of word vectors
    variantName = "variant5" #choose the variant type

    padded_docs, Y, t = buildInput(max_length, inputFile)
    embedding_matrix = getEmbeddingMatrix(t, embeddingDim)
    test(max_length, embeddingDim, t, padded_docs, Y, embedding_matrix, variantName)


if __name__ == '__main__':
    main()
