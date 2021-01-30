import numpy as np
import pandas as pd
from skimage import io
from dct import dct2
from mask import mask
import time

if __name__ == '__main__':
    df = pd.DataFrame()
    np.set_printoptions(linewidth=200, precision=3, suppress=True)
    image = np.arange(1024*1024).reshape((1024,1024))
    print(image.shape)
    image = dct2(image)
    loop = 1
    for j in range(1,11):
        for i in range(image.shape[0]//8//10*(j-1)+1,image.shape[0]//8//10*j,j):
            avg = 0
            for _ in range(loop):
                start = time.time()
                a = mask(image,4,8*i+4,4,8*i+4)
                end = time.time()
                avg += (end-start)/loop
            # print('{}x{}:{}[s]'.format(8*i,8*i,avg))
            df = df.append({'size':i*8,'time':avg},ignore_index=True)
    df.to_csv('result.csv',index=False)