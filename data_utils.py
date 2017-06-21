import sys
import config
from six.moves import cPickle as pickle
import glob
import random
import numpy as np
import scipy.misc

def load_PROTEST(d_id=0,filename=None, batches=False, larger=False, features=False):
    if filename is None:
        if batches:
            if larger:
                if features:
                    filename = "%s%s_batch.save" % (config.VGG_FEATURES,d_id)
                else:
                    filename = "%s%s_batch.save" % (config.ALL_DATA_224,d_id)
            else:
                filename = "%s%s_batch.save" % (config.ALL_DATA,d_id)
        else:
            filename = "%s%s_dataset.save" % (config.DATASETS,d_id)
    try:
        f = open(filename, 'rb')
    except:
        print("Error opening: %s" % filename)
        return 0
    else:
        data = pickle.load(f)
        f.close()
    return data

def load_TARGET(resized=False, features=False):
    filename = "%starget_dataset.save" % (config.DATASETS)
    if resized:
        filename = "%starget_224_dataset.save" % (config.DATASETS)
    if features:
        filename = "%starget_224_dataset.save" % (config.VGG_FEATURES)
    f = open(filename, 'rb')
    data = pickle.load(f)
    f.close()
    return data

def save_data(data, d_id=0, all_data=False):
    filename = None
    if all_data:
        filename = "%s%s_batch.save" % (config.ALL_DATA,d_id)
    else:
        filename = "%s%s_dataset.save" % (config.DATASETS,d_id)
        
    f = open(filename, 'wb')
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()
    
#creates the train set and val set for files in filenames. Returns lists
def create_set(ntrain, nval, filenames):    
    train_set = list()
    for i in range(ntrain):
        train_set.append(scipy.misc.imread(filenames[i]))
        if i % 100 == 0:
            print("finished %d of %d" % (i,ntrain))
    print("finished a trainset")
    val_set = [scipy.misc.imread(filenames[ntrain+j]) for j in range(nval)]
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


#creates a batch dataset for files in filenames. Returns lists
def create_batch_set(filenames, verbose=False):    
    train_set = list()
    for i in range(len(filenames)):
        train_set.append(scipy.misc.imread(filenames[i]))
        if i % 100 == 0 and verbose:
            print("finished %d of %d" % (i,ntrain))
    return train_set


def parse_all_data():

    batchsize = 5000
    max_images = 220000
    
    random.seed(17)
    neg_files = glob.glob(config.NEGATIVE_SAMPLES+"*.png")
    pos_files = glob.glob(config.POSITIVE_SAMPLES+"*.png")
    random.shuffle(neg_files)
    random.shuffle(pos_files)

    print(len(neg_files))
    print(len(pos_files))

    num_batches = int(max_images/batchsize)
    
    print("Starting batch processing")
    print("%d total batches" % num_batches)
    
    for batch in range(10,num_batches):
    
        start_idx = batch*batchsize
        end_idx = (batch+1)*batchsize
        
        pos_samples = create_batch_set(pos_files[start_idx:end_idx])
        neg_samples = create_batch_set(neg_files[start_idx:end_idx])

        #lets mix them up and create a labels array
        X_samples, y_samples = getLabeled(pos_samples, neg_samples)

        # Data sets above are still lists. Here we STACK them into NP arrays and store
        # them in data dict()
        data = dict()
        data["X_train"]=np.stack(X_samples)
        data["y_train"]=np.array(y_samples)

        # save the data dict so it can be loaded using load_PROTEST_ALL
        save_data(data,batch,True)

        print("Finished batch %d" % batch)  

def main():

    parse_all_data()
    exit()
    
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
