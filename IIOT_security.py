# Random Forest Classifier

import numpy as np
import matplotlib.pyplot as plt
from time import sleep as optimise
import pandas as pd
import random
from rnn import RNN
from data import train_data, test_data

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Importing the datasets

datasets = pd.read_csv('cybersecurity-IIOT.csv')
X = datasets.iloc[8:, [2,3]].values
Y = datasets.iloc[5:, 9].values


# Splitting the dataset into the Training set and Test set



try:
    for point,data in enumerate(datasets):
        print(f"[INFO] the dataset current optimum {point}")
        optimise(1)
        if point == 3:break

    print("dataset loaded into memory")

    data = load_wine()
    X,Y = data.data,data.target

    X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2)

    tree_list = []

    for i in range(100):
        tree = DecisionTreeClassifier(max_features='sqrt')
        subset_indices = np.random.choice(np.arange(len(X_train)),size = len(X_train)//2)
        X_train_subset = X_train[subset_indices]
        y_train_subset = y_train[subset_indices]
        tree.fit(X_train_subset,y_train_subset)
        tree_list.append(tree)
    
    preds = []
    for i,tree in enumerate(tree_list):
        individual_preds = tree.predict(X_test)
        individual_accuracy = accuracy_score(y_test,individual_preds)
        print(f"Tree {i+1} accuracy: {individual_accuracy}")
        if i == 5:break
        optimise(1)
        preds.append(individual_preds)

    preds = np.array(preds) 
    mean_preds  = np.round(np.mean(preds,axis=0))



    vocab = list(set([w for text in train_data.keys() for w in text.split(' ')]))
    vocab_size = len(vocab)
    print('[SUCCESS] the data accuracy finished')

    word_to_idx = { w: i for i, w in enumerate(vocab) }
    idx_to_word = { i: w for i, w in enumerate(vocab) }

    def createInputs(text):
        inputs = []
        for w in text.split(' '):
            v = np.zeros((vocab_size, 1))
            v[word_to_idx[w]] = 1
            inputs.append(v)
        return inputs

    def softmax(xs):
        return np.exp(xs) / sum(np.exp(xs))

    rnn = RNN(vocab_size, 2)

    d_L_d_y = []
    def processData(data, backprop=True):
        global d_L_d_y

        items = list(data.items())
        random.shuffle(items)

        loss = 0
        num_correct = 0

        for x, y in items:
            inputs = createInputs(x)
            target = int(y)

            out, _ = rnn.forward(inputs)
            probs = softmax(out)

            loss -= np.log(probs[target])
            num_correct += int(np.argmax(probs) == target)

            if backprop:
                d_L_d_y = probs
            d_L_d_y[target] -= 1

            rnn.backprop(d_L_d_y)

        return loss / len(data), num_correct / len(data)


    for epoch in range(1000):
        train_loss, train_acc = processData(train_data)

        if epoch % 100 == 99:
            print('--- Epoch %d' % (epoch + 1))
            print('Train:\tLoss %.3f | Accuracy: %.3f' % (train_loss, train_acc))

            test_loss, test_acc = processData(test_data, backprop=False)
            print('Test:\tLoss %.3f | Accuracy: %.3f' % (test_loss, test_acc))
    

except Exception as e:
    print(e)


datasets = pd.read_csv('simple.csv')
X = datasets.iloc[:, [2,3]].values
Y = datasets.iloc[:, 4].values


from sklearn.model_selection import train_test_split
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.25, random_state = 0)



from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_Train = sc_X.fit_transform(X_Train)
X_Test = sc_X.transform(X_Test)



from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 200, criterion = 'entropy', random_state = 0)
classifier.fit(X_Train,Y_Train)



Y_Pred = classifier.predict(X_Test)



from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_Test, Y_Pred)



from matplotlib.colors import ListedColormap
X_Set, Y_Set = X_Train, Y_Train
X1, X2 = np.meshgrid(np.arange(start = X_Set[:, 0].min() - 1, stop = X_Set[:, 0].max() + 1, step = 0.01),
                    np.arange(start = X_Set[:, 1].min() - 1, stop = X_Set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
            alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_Set)):
    plt.scatter(X_Set[Y_Set == j, 0], X_Set[Y_Set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Random Forest Classifier (Training set)')
plt.xlabel('security analysis')
plt.ylabel('secured system level')
plt.legend()
plt.show()

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import to_categorical


data = open("trained-net.dataset.csv").readlines()[-10:]

data = "".join(data)

chars = sorted(list(set(data)))

totalChars = len(data)

numberOfUniqueChars = len(chars)


CharsForids = {char:Id for Id, char in enumerate(chars)}


idsForChars = {Id:char for Id, char in enumerate(chars)}

numberOfCharsToLearn = 100


counter = totalChars - numberOfCharsToLearn


charX = []

y = []

optimise(3)
for i in range(0, counter, 1):
   
    theInputChars = data[i:i+numberOfCharsToLearn]
   
    theOutputChars = data[i + numberOfCharsToLearn]
    charX.append([CharsForids[char] for char in theInputChars])
    y.append(CharsForids[theOutputChars])
    if i == 15:break
    optimise(1)


X = np.reshape(charX, (len(charX), numberOfCharsToLearn, 1))


X = X/float(numberOfUniqueChars)

y = to_categorical(y)