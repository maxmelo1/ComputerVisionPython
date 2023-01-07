import numpy as np
import matplotlib.image as mpimg
import os
from imutils import paths
from skimage.color import rgb2gray
import skimage.feature as ft
import csv
import pandas as pd



class NearestNeighbor:
    def __init__(self):
        pass

    def train(self, X, y):
        self.Xtr = X
        self.ytr = y

    def predict(self, X):
        """ X é NxD onde cada linha é um exemplo que queremos predizer o label"""
        num_test = X.shape[0]

        print("%d amostras"%(X.shape[0]) )

        #criando o tipo
        #Ypred = np.zeros(num_test, dtype = self.ytr.dtype)
        Ypred = np.zeros(num_test, dtype = np.int)

        ind = []

        for i in range(num_test):
            #print("xtr %d x %d " %(len(self.Xtr), len(X) ) )
            distances = np.sum(np.abs(self.Xtr - X[i, :]), axis = 1)
            min_index = np.argmin(distances) #pegar o menor
            Ypred[i]  = self.ytr[min_index] #predizer o label do exemplo mais próximo

            ind.append(i+1)

        ret = {
            "id": ind,
            "label": Ypred
        }

        #print(ret)

        return ret


def main():

    trainImgs   = []
    testImgs    = []

    trainLabels = []
    testLabels  = []

    with open( '../gencsv/lbp_train.csv', mode='r' ) as file:
        reader = csv.reader(file, delimiter=',')

        for row in reader:
            trainImgs.append( row[0:22] )
            trainLabels.append( row[-1] )

    with open( '../gencsv/lbp_test.csv', mode='r' ) as file:
        reader = csv.reader(file, delimiter=',')

        for row in reader:
            testImgs.append( row[0:22] )
            testLabels.append( row[-1] )


    cl = NearestNeighbor()

    #test = np.array(trainImgs, dtype=np.float32)

    cl.train( np.array(trainImgs, dtype=np.float32), np.array(trainLabels, dtype=np.float32) )

    y = cl.predict( np.array(testImgs, dtype=np.float32) )

    acc = 0

    print(len(y))

    df = pd.DataFrame(y)
    df.to_csv("res.csv", index = None, header=True)


if __name__ == '__main__':
    main()
