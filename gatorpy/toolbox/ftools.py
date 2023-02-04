from os.path import *
import pickle


def pathjoin(p,ne): # '/path/to/folder', 'name.extension' (or a subfolder)
    return join(p,ne)

def saveData(data,path):
    print('saving data')
    dataFile = open(path, 'wb')
    pickle.dump(data, dataFile)

def loadData(path):
    print('loading data')
    dataFile = open(path, 'rb')
    return pickle.load(dataFile)