from argparse import ArgumentParser
import numpy as np
from PIL import Image
from utils import *
import os
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
# Test script

def load_data(input_file):
    d = unpickle(input_file)
    x = d['data']
    y = d['labels']
    x = np.dstack((x[:, :1024], x[:, 1024:2048], x[:, 2048:]))
    x = x.reshape((x.shape[0], 32, 32, 3))
    return x, y

if __name__ == '__main__':

    batches = [1,2,3,4,5,6,7,8,9,10]
    input_path = "../Imagenet32_train/train_data_batch_"
 
    for batch in batches:
        x, y = load_data(input_path+str(batch))
        curr_index = 0
        image_index = 0

        print('First image in dataset:')
        print(x[curr_index])

        if not os.path.exists('res'):
            os.makedirs('res')

        for i in range(x.shape[0]):
            Image.fromarray(x[curr_index]).save('res/batch_'+str(batch)+'_Image_'+str(curr_index)+'.png')
            curr_index += 1


