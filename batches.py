import sys
import config
from six.moves import cPickle as pickle
import glob
import random
import numpy as np
import scipy.misc

def load_PROTEST(d_id=0):
    filename = "%s%d_dataset.save" % (config.DATASETS,d_id)
    f = open(filename, 'rb')
    data = pickle.load(f)
    f.close()
    return data


def load_TARGET():
    filename = "%starget_dataset.save" % (config.DATASETS)
    f = open(filename, 'rb')
    data = pickle.load(f)
    f.close()
    return data
    

def save_data(data, d_id=0):
    filename = "%s%d_dataset.save" % (config.DATASETS,d_id)
    f = open(filename, 'wb')
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()
    
#creates the train set and val set for files in filenames. Returns lists
def create_set(off, ntrain, nval, filenames):    
    train_set = list()
    for i in range(ntrain):
        train_set.append(scipy.misc.imread(filenames[off+i]))
        if i % 100 == 0:
            print("finished %d of %d" % (i,ntrain))
    print("finished a trainset")
    val_set = [scipy.misc.imread(filenames[off+ntrain+j]) for j in range(nval)]
    print("finished a valset")
    return train_set, val_set

#mixes up the positive and negative train samples and assigns labels
def getLabeled(d_pos, d_neg):
    X = d_pos + d_neg
    y = [config.pos_label]*len(d_pos) + [config.neg_label]*len(d_neg)
    Z = list(zip(X, y))
    random.shuffle(Z)
    X, y = zip(*Z)
    return X, y
        
#generates a serial data object that can be loaded using load_data
def generate_protest_dataset(N,V,d_id):

    random.seed(17)
    neg_files = glob.glob(config.NEGATIVE_SAMPLES+"*.png")
    pos_files = glob.glob(config.POSITIVE_SAMPLES+"*.png")
    random.shuffle(neg_files)
    random.shuffle(pos_files)

    #take files
    num_pos_train = int(N/2.0)
    num_neg_train = N-num_pos_train
    num_pos_val = int(V/2.0)
    num_neg_val = V-num_pos_val
    
    assert(num_neg_train>0 and num_neg_val>0)
    
    pos_train, pos_val = create_set(num_pos_train, num_pos_val, pos_files)
    neg_train, neg_val = create_set(num_neg_train, num_neg_val, neg_files)

    #lets mix them up and create a labels array
    X_train, y_train = getLabeled(pos_train, neg_train)
    X_val, y_val = getLabeled(pos_val, neg_val)

    # Data sets above are still lists. Here we STACK them into NP arrays and store
    # them in data dict()
    data = dict()
    data["X_train"]=np.stack(X_train)
    data["y_train"]=np.array(y_train)
    data["X_val"]=np.stack(X_val)
    data["y_val"]=np.array(y_val)

    # save the data dict so it can be loaded using load_PROTEST
    save_data(data,d_id)
    
    '''
    # TESTING PURPOSES. SAVES 3 IMAGES FROM data["X_train"]
    for i in range(3):
        print(data["X_train"][i])
        scipy.misc.imsave(str(i)+".png", data["X_train"][i])
    '''

    return (num_pos_train, num_neg_train, num_pos_val, num_neg_val)
    


def main():
    print("MAKE SURE TO USE PYTHON 3\n")

    new_batch = input("Would you like to create a new batch of train data? [y/n] ")

    if new_batch.lower() != "y":
        return        

    N = int(input("Number of training samples? "))
    V = int(input("Number of validation samples? "))

    assert(N>0 and V>0)

    d_id = int(input("Provide and integer id number for this dataset: [default=0] "))
    
    nposTr, nnegTr, nposVal, nnegVal = generate_protest_dataset(N,V,d_id)

    print("Dataset generated with %d pos and %d neg train samples" % (nposTr, nnegTr))
    print("Dataset generated with %d pos and %d neg val samples" % (nposVal, nnegVal))

    print("dataset saved at %s/%d_dataset.save" % (config.DATASETS,d_id))
    
if __name__ == "__main__":
        main()
