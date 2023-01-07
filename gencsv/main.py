from imutils import paths
from skimage.color import rgb2gray
import matplotlib.image as mpimg
import skimage.feature as ft
import csv
import pandas as pd
import numpy as np
import time

#https://medium.com/@burhanahmeed/image-recognition-with-svm-and-local-binary-pattern-289cc19ba7fe

class GenCSV:
    # settings for LBP
    METHOD = 'uniform'
    P = 16
    R = 20

    def __init__(self):
        pass

    def generateLBP(self, csvName, p, eps=1e-7):
        trainPaths = list(paths.list_images(p))

        trainLabels = []
        trainImgs   = []



        #because of mem size we need to partition the writing
        cont    = 0
        written = 0

        size = len(trainPaths)

        vet = []

        for path in trainPaths:


            str = path.split("/")
            #print(str[4])
            label =  1 if "dog" in str[4] else 0

            img  = rgb2gray(mpimg.imread(path))
            lbp = ft.local_binary_pattern(img, self.P, self.R, self.METHOD)

            (hist, _) = np.histogram(lbp.ravel(),
            bins=np.arange(0, self.R + 3),
            range=(0, self.R + 2))

            hist  = hist.astype("float")
            hist /= (hist.sum() + eps)

            #print(hist)
            aux = list(hist)

            #tere = np.append(tere, label)
            aux.append(label)

            #print(aux)

            #nova linha
            vet.append(aux)

            #print("adfas")
            #print(aux)

            cont = cont + 1

            if cont > 2000:
                df = pd.DataFrame(vet)
                if written == 0:
                    #print(vet)
                    df.to_csv(csvName, index = None, header=False)
                elif written > 0:
                    with open(csvName, 'a') as f:
                        df.to_csv(f, index = None, header=False)

                #print("nova linha")
                #print(len(vet))

                #time.sleep(5)

                cont = 0
                written  = written + 1
                vet = []
                print("%.2f  concluído" % ((2000*written)/size))

def main():
    obj = GenCSV()
    obj.generateLBP('lbp_train.csv', "../knn/dogs-vs-cats/train")
    obj.generateLBP('lbp_test.csv', "../knn/dogs-vs-cats/test1")



if __name__ == '__main__':
    main()
