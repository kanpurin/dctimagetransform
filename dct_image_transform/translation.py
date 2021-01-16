import numpy as np
from dct_image_transform.dct import dct2
from dct_image_transform.reflection import reflection

__all__ = ['translation']

def _translation(image,sigma):
    im = image.copy()
    if im.shape[1] <= np.abs(sigma):
        im[:] = 0
        return im
    im[:,sigma//8*8:] = im[:,:im.shape[1]-sigma//8*8]
    im[:,:sigma//8*8] = 0
    sigma = sigma % 8

    # Vの定義
    V1 = np.zeros((8,8),dtype=np.float) # V[sigma]
    V2 = np.zeros((8,8),dtype=np.float) # V[w-sigma]
    for y in range(8-sigma):
        V1[y,sigma+y] = 1
    V1 = dct2(V1)
    for y in range(sigma):
        V2[8-sigma+y,y] = 1
    V2 = dct2(V2)
    
    im[:,:8] = np.dot(im[:,:8],V1)
    for j in range(im.shape[1]//8-1,0,-1):
        im[:,j*8:j*8+8] = np.dot(im[:,j*8:j*8+8],V1)+np.dot(im[:,j*8-8:j*8],V2)
    return im

def translation(image,sigma,axis=0):
    '''
    8x8のブロックごとに離散コサイン変換された画像(以下DCT画像)を平行移動する.

    Parameters
    ----------
    image:幅と高さが8の倍数であるDCT画像を表す2次元配列. 8の倍数でない場合の動作は未定義.
    
    sigma:移動距離. 負数も可(ただし, 少し遅い)

    axis:移動方向の軸. defaultは`axis=0`

    Returns
    -------
    `image`を平行移動したDCT画像を表す2次元配列を返す. `image`の値は変わらない.

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
    >>> dct_image_transform.translation.translation(a,4,axis=1)
    array([[  0,   1,  -2,   0,   3,   0,  -7,   8],
           [  6,   0, -11,   3,  10,  -5, -16,  22],
           [ 12,  -1, -19,   7,  17, -10, -25,  36],
           [ 19,  -3, -28,  11,  25, -16, -34,  50],
           [ 25,  -4, -36,  16,  32, -22, -43,  64],
           [ 32,  -6, -45,  20,  39, -27, -52,  79],
           [ 38,  -7, -53,  24,  47, -33, -61,  93],
           [ 45,  -9, -62,  28,  54, -38, -71, 107]])
    '''

    if sigma < 0:
        sigma = -sigma
        if axis == 0:
            return np.transpose(reflection(_translation(reflection(np.transpose(image),axis=1), sigma),axis=1))
        elif axis == 1:
            return reflection(_translation(reflection(image,axis=1), sigma),axis=1)
    else:
        if axis == 0:
            return np.transpose(_translation(np.transpose(image), sigma))
        elif axis == 1:
            return _translation(image, sigma)