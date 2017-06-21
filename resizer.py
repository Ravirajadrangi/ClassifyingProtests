from data_utils import load_PROTEST
from data_utils import load_TARGET
from data_utils import save_data
import scipy.misc
import numpy as np
import glob
from six.moves import cPickle as pickle
import config


def resize_arr(data_arr,H,W):
    N,_,_,D = data_arr.shape

    new_arr = list()
    for i in range(N):
        new_arr.append(scipy.misc.imresize(data_arr[i], (H,W), interp='bilinear'))    
    arr = np.stack(new_arr,axis=0)
    return arr

def resize_data(data, size_tuple):
    H, W = size_tuple
    new_data = dict()
    #data is a hash map
    for key in data:
        if key[0]=='y':
            new_data[key] = data[key]
        else:
            new_data[key] = resize_arr(data[key],H,W)
        
    return new_data

#loop through batches to resize
def convert_all_data():
    batch_files = glob.glob(config.ALL_DATA+"*.save")
    for batch_file in batch_files:
        raw_batch = load_PROTEST(filename=batch_file)
        resized_batch = resize_data(raw_batch, (224,224))
        suffix = batch_file.split("/")[-1]
        new_filename = "%s%s" % (config.ALL_DATA_224,suffix)
        f = open(new_filename, 'wb')
        pickle.dump(resized_batch, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()
        exit()


#loop through batches to resize
def convert_all_data_by_filenames(count=33):
    prefix = "/DATA/SHARE/datasets/all_data/"
    suffix = "_batch.save"

    batchno = 11
    num_converted = 0
    
    while num_converted < count:
        batch_file = "%s%d%s" % (prefix,batchno,suffix)
        #skip number 9
        if batchno == 9:
            batchno += 1
            continue
        raw_batch = load_PROTEST(filename=batch_file)
        resized_batch = resize_data(raw_batch, (224,224))
        new_filename = "%s%d%s" % (config.ALL_DATA_224,batchno,suffix)
        f = open(new_filename, 'wb')
        pickle.dump(resized_batch, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()

        print("Converted batch number: %d" % batchno)
        num_converted += 1
        batchno += 1
        
convert_all_data_by_filenames()
    

#raw_data=load_PROTEST(20)
#raw_data = load_TARGET()
#new_data = resize_data(raw_data, (224, 224))
#save_data(new_data,"20_224")

