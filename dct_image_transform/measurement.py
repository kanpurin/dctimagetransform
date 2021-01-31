import numpy as np
import pandas as pd
from dct import dct2,idct2
from mask import mask
import time
import sys

if __name__ == '__main__':
    loop = 1000 #ループ回数
    runtype = 0 #0:等間隔, other:徐々に広がる
    output = 0 #出力するか
    if len(sys.argv) >= 2:
        loop = int(sys.argv[1]) # 回数
    if len(sys.argv) >= 3:
        runtype = int(sys.argv[2])
    if len(sys.argv) >= 4:
        output = int(sys.argv[3])
    df = pd.DataFrame()
    np.set_printoptions(linewidth=200, precision=3, suppress=True)
    image_size = 1024
    image = np.random.randint(0, 255, (image_size, image_size))
    image = dct2(image)
    if runtype == 0:
        for i in range(1,image.shape[0]//8):
            avg = 0
            for _ in range(loop):
                start = time.time()
                a = idct2(image)
                # a = mask(image,4,8*i+4,4,8*i+4)
                a[4:8*i+4,4:8*i+4] = 0
                a = dct2(a)
                end = time.time()
                avg += (end-start)/loop
            if output:
                print('{}x{}:{}[s]'.format(8*i,8*i,avg))
            df = df.append({'size':i*8,'time':avg},ignore_index=True)
    else:
        for j in range(1,11):
            for i in range(image.shape[0]//8//10*(j-1)+1,image.shape[0]//8//10*j,j):
                avg = 0
                for _ in range(loop):
                    start = time.time()
                    a = mask(image,4,8*i+4,4,8*i+4)
                    end = time.time()
                    avg += (end-start)/loop
                if output:
                    print('{}x{}:{}[s]'.format(8*i,8*i,avg))
                df = df.append({'size':i*8,'time':avg},ignore_index=True)
    df.to_csv('result_mask_2.csv',index=False)