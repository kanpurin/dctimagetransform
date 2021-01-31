import numpy as np
import pandas as pd
from dct import dct2,idct2
from mask import mask
from translation import translation
from reflection import reflection
from rotate import rotate90, rotate180
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
    np.set_printoptions(linewidth=200, precision=3, suppress=True)
    image_size = 1024
    image = np.random.randint(0, 255, (image_size, image_size))
    image = dct2(image)
    df = pd.DataFrame()
    if runtype == 0:
        # 平行移動(直接)
        avg = 0
        for i in range(loop):
            start = time.time()
            a = translation(image,4,axis=1)
            end = time.time()
            avg += (end-start)/loop
        if output:
            print('translation1:{}[s]'.format(avg))
        df = df.append({'name':'translation1','time':avg},ignore_index=True)

        # 平行移動(通常)
        avg = 0
        for i in range(loop):
            start = time.time()
            a = idct2(image)
            a[:,4:] = a[:,:image_size-4]
            a[:,:4] = 0
            a = dct2(a)
            end = time.time()
            avg += (end-start)/loop
        if output:
            print('translation2:{}[s]'.format(avg))
        df = df.append({'name':'translation2','time':avg},ignore_index=True)

        # 鏡像変換(直接)
        avg = 0
        for i in range(loop):
            start = time.time()
            a = reflection(image,axis=1)
            end = time.time()
            avg += (end-start)/loop
        if output:
            print('reflection1:{}[s]'.format(avg))
        df = df.append({'name':'reflection1','time':avg},ignore_index=True)

        # 鏡像変換(通常)
        avg = 0
        for i in range(loop):
            start = time.time()
            a = idct2(image)
            a = a[:,::-1]
            a = dct2(a)
            end = time.time()
            avg += (end-start)/loop
        if output:
            print('reflection2:{}[s]'.format(avg))
        df = df.append({'name':'reflection2','time':avg},ignore_index=True)

        # 90度回転(直接)
        avg = 0
        for i in range(loop):
            start = time.time()
            a = rotate90(image)
            end = time.time()
            avg += (end-start)/loop
        if output:
            print('rotate901:{}[s]'.format(avg))
        df = df.append({'name':'rotate901','time':avg},ignore_index=True)

        # 90度回転(通常)
        avg = 0
        for i in range(loop):
            start = time.time()
            a = idct2(image)
            a = np.rot90(a)
            a = dct2(a)
            end = time.time()
            avg += (end-start)/loop
        if output:
            print('rotate902:{}[s]'.format(avg))
        df = df.append({'name':'rotate902','time':avg},ignore_index=True)

        # 180度回転(直接)
        avg = 0
        for i in range(loop):
            start = time.time()
            a = rotate180(image)
            end = time.time()
            avg += (end-start)/loop
        if output:
            print('rotate1801:{}[s]'.format(avg))
        df = df.append({'name':'rotate1801','time':avg},ignore_index=True)

        # 180度回転(通常)
        avg = 0
        for i in range(loop):
            start = time.time()
            a = idct2(image)
            a = np.rot90(a,k=2)
            a = dct2(a)
            end = time.time()
            avg += (end-start)/loop
        if output:
            print('rotate1802:{}[s]'.format(avg))
        df = df.append({'name':'rotate1802','time':avg},ignore_index=True)

        # for i in range(1,image.shape[0]//8):
        #     avg = 0
        #     for _ in range(loop):
        #         start = time.time()
        #         a = idct2(image)
        #         # a = mask(image,4,8*i+4,4,8*i+4)
        #         a[4:8*i+4,4:8*i+4] = 0
        #         a = dct2(a)
        #         end = time.time()
        #         avg += (end-start)/loop
        #     if output:
        #         print('{}x{}:{}[s]'.format(8*i,8*i,avg))
        #     df = df.append({'size':i*8,'time':avg},ignore_index=True)
    # else:
        # for j in range(1,11):
        #     for i in range(image.shape[0]//8//10*(j-1)+1,image.shape[0]//8//10*j,j):
        #         avg = 0
        #         for _ in range(loop):
        #             start = time.time()
        #             a = mask(image,4,8*i+4,4,8*i+4)
        #             end = time.time()
        #             avg += (end-start)/loop
        #         if output:
        #             print('{}x{}:{}[s]'.format(8*i,8*i,avg))
        #         df = df.append({'size':i*8,'time':avg},ignore_index=True)
    # print(df)
    df.to_csv('result_other.csv',index=False)