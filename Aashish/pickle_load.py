import numpy as np
import cPickle as pickle
import os

def load_CIFAR_batch(filename, meta = False):
  """ load single batch of cifar """
  with open(filename, 'rb') as f:
    datadict = pickle.load(f)
    if not meta:
      X = datadict['data']
      Y = datadict['labels']
      X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float") #reshaping the image matrix changing the axes 
      Y = np.array(Y)
      return X, Y
    else: 
      return datadict

def load_CIFAR10(ROOT, one_hot = True):
  """ load all of cifar """
  xs = []
  ys = []
  for b in range(1,6):
    f = os.path.join(ROOT, 'data_batch_%d' % (b))
    X, Y = load_CIFAR_batch(f)
    xs.append(X)
    ys.append(Y)    
    del X, Y
  Xtr = np.concatenate(xs) #loads training image
  Ytr = np.concatenate(ys) #loads training labels
  del xs , ys
  meta_batches = load_CIFAR_batch(os.path.join(ROOT, 'batches.meta'), meta = True)
  labels = meta_batches['label_names']
  del meta_batches
  Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch')) #loads test images and test labels
  if one_hot:
      Y = np.zeros((len(Ytr), 10))
      for y1,y2 in zip(range(0,len(Ytr)),Ytr):
        Y[y1][y2] = 1
      del y1,y2
      return Xtr, Ytr, Y, Xte, Yte, labels
  return Xtr, Ytr, Xte, Yte, labels
Xtr, Ytr, Y, Xte, Yte, labels = load_CIFAR10('/home/aashish/Documents/cifar-10-batches-py') # a magic function we provide
print Y

