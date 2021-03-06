import numpy as np
from dct_image_transform.dct import dct2

def reflection(image,axis=0):
    '''
    8x8のブロックごとに離散コサイン変換された画像(以下DCT画像)を鏡像変換する.

    Parameters
    ----------
    image:幅と高さが8の倍数である画像を表す2次元配列. 8の倍数でない場合の動作は未定義.
    
    axis:変換する軸. defalutは`axis=0`

    Returns
    -------
    `image`を鏡像変換したDCT画像を表す2次元配列を返す. `image`の値は変わらない.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.arange(64).reshape((8,8))
    >>> a
    array([[ 0,  1,  2,  3,  4,  5,  6,  7],
           [ 8,  9, 10, 11, 12, 13, 14, 15],
           [16, 17, 18, 19, 20, 21, 22, 23],
           [24, 25, 26, 27, 28, 29, 30, 31],
           [32, 33, 34, 35, 36, 37, 38, 39],
           [40, 41, 42, 43, 44, 45, 46, 47],
           [48, 49, 50, 51, 52, 53, 54, 55],
           [56, 57, 58, 59, 60, 61, 62, 63]])
    >>> dct_image_transform.reflection.reflection(a,axis=0)
    array([[ 5.77395663e-15,  1.00000000e+00,  2.00000000e+00,
             3.00000000e+00,  4.00000000e+00,  5.00000000e+00,
             6.00000000e+00,  7.00000000e+00],
           [-8.00000000e+00, -9.00000000e+00, -1.00000000e+01,
            -1.10000000e+01, -1.20000000e+01, -1.30000000e+01,
            -1.40000000e+01, -1.50000000e+01],
           [ 1.60000000e+01,  1.70000000e+01,  1.80000000e+01,
             1.90000000e+01,  2.00000000e+01,  2.10000000e+01,
             2.20000000e+01,  2.30000000e+01],
           [-2.40000000e+01, -2.50000000e+01, -2.60000000e+01,
            -2.70000000e+01, -2.80000000e+01, -2.90000000e+01,
            -3.00000000e+01, -3.10000000e+01],
           [ 3.20000000e+01,  3.30000000e+01,  3.40000000e+01,
             3.50000000e+01,  3.60000000e+01,  3.70000000e+01,
             3.80000000e+01,  3.90000000e+01],
           [-4.00000000e+01, -4.10000000e+01, -4.20000000e+01,
            -4.30000000e+01, -4.40000000e+01, -4.50000000e+01,
            -4.60000000e+01, -4.70000000e+01],
           [ 4.80000000e+01,  4.90000000e+01,  5.00000000e+01,
             5.10000000e+01,  5.20000000e+01,  5.30000000e+01,
             5.40000000e+01,  5.50000000e+01],
           [-5.60000000e+01, -5.70000000e+01, -5.80000000e+01,
            -5.90000000e+01, -6.00000000e+01, -6.10000000e+01,
            -6.20000000e+01, -6.30000000e+01]])
    '''
    R = np.zeros((8,8),dtype=np.float)
    for i in range(8):
        R[i,7-i] = 1
    R = dct2(R)
    if axis == 0:
        return np.vstack(list(map(lambda m:np.dot(R,m),np.flip(np.vsplit(image,range(8,image.shape[1],8)),0))))
    elif axis == 1:
        return np.hstack(list(map(lambda m:np.dot(m,R),np.flip(np.hsplit(image,range(8,image.shape[1],8)),0))))