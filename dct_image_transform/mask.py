from dct_image_transform.dct import dct2, idct2
import numpy as np

def mask(image,ulx,lrx,uly,lry):
    '''
    8x8のブロックごとに離散コサイン変換された画像(以下DCT画像)の一部をマスクする.

    Parameters
    ----------
    image:幅と高さが8の倍数である画像を表す2次元配列. 8の倍数でない場合の動作は未定義.
    
    image[ulx:lrx,uly:lry]をマスクする.

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
    >>> dct_image_transform.mask.mask(a,3,6,2,4)
    array([[ 0,  2,  2,  1,  2,  4,  7,  8],
           [ 7,  9, 10, 10, 11, 13, 13, 14],
           [15, 15, 17, 20, 22, 21, 19, 21],
           [23, 24, 25, 27, 29, 29, 28, 28],
           [31, 33, 33, 34, 35, 37, 38, 38],
           [40, 43, 41, 39, 40, 44, 50, 50],
           [48, 49, 49, 50, 50, 53, 54, 55],
           [52, 55, 56, 64, 64, 61, 55, 52]])
    '''
    f = image.copy()
    _ulx = ulx//8
    _lrx = (lrx-1)//8
    _uly = uly//8
    _lry = (lry-1)//8
    if _ulx != _lrx and _uly != _lry:
        # 左上
        f[_ulx*8:_ulx*8+8,_uly*8:_uly*8+8] = idct2(f[_ulx*8:_ulx*8+8,_uly*8:_uly*8+8])
        f[ulx:_ulx*8+8,uly:_uly*8+8] = 0
        f[_ulx*8:_ulx*8+8,_uly*8:_uly*8+8] = dct2(f[_ulx*8:_ulx*8+8,_uly*8:_uly*8+8])
        
        # 右上
        f[_ulx*8:_ulx*8+8,_lry*8:_lry*8+8] = idct2(f[_ulx*8:_ulx*8+8,_lry*8:_lry*8+8])
        f[ulx:_ulx*8+8,_lry*8:lry] = 0
        f[_ulx*8:_ulx*8+8,_lry*8:_lry*8+8] = dct2(f[_ulx*8:_ulx*8+8,_lry*8:_lry*8+8])
        
        # 左下
        f[_lrx*8:_lrx*8+8,_uly*8:_uly*8+8] = idct2(f[_lrx*8:_lrx*8+8,_uly*8:_uly*8+8])
        f[_lrx*8:lrx,uly:_uly*8+8] = 0
        f[_lrx*8:_lrx*8+8,_uly*8:_uly*8+8] = dct2(f[_lrx*8:_lrx*8+8,_uly*8:_uly*8+8])
        
        # 右下
        f[_lrx*8:_lrx*8+8,_lry*8:_lry*8+8] = idct2(f[_lrx*8:_lrx*8+8,_lry*8:_lry*8+8])
        f[_lrx*8:lrx,_lry*8:lry] = 0
        f[_lrx*8:_lrx*8+8,_lry*8:_lry*8+8] = dct2(f[_lrx*8:_lrx*8+8,_lry*8:_lry*8+8])

        # 上
        if ulx%8 == 0:
            f[_ulx*8:_ulx*8+8,_uly*8+8:_lry*8] = 0
        else:
            M = np.zeros((8,8))
            for i in range(ulx%8):
                M[i,i] = 1
            M = dct2(M)
            f[_ulx*8:_ulx*8+8,(_uly+1)*8:_lry*8] = np.dot(M,f[_ulx*8:_ulx*8+8,(_uly+1)*8:_lry*8])

        # 左
        if uly%8 == 0:
            f[_ulx*8+8:_lrx*8,_uly*8:_uly*8+8] = 0
        else:
            M = np.zeros((8,8))
            for i in range(uly%8):
                M[i,i] = 1
            M = dct2(M)
            f[(_ulx+1)*8:_lrx*8,_uly*8:_uly*8+8] = np.dot(f[(_ulx+1)*8:_lrx*8,_uly*8:_uly*8+8],M)
                
        # 右
        if lry%8 == 0:
            f[_ulx*8+8:_lrx*8,_lry*8:_lry*8+8] = 0
        else:
            M = np.zeros((8,8))
            for i in range(lry%8,8):
                M[i,i] = 1
            M = dct2(M)
            f[(_ulx+1)*8:_lrx*8,_lry*8:_lry*8+8] = np.dot(f[(_ulx+1)*8:_lrx*8,_lry*8:_lry*8+8],M)

        # 下
        if lrx%8 == 0:
            f[_lrx*8:_lrx*8+8,_uly*8+8:_lry*8] = 0
        else:
            M = np.zeros((8,8))
            for i in range(lrx%8,8):
                M[i,i] = 1
            M = dct2(M)
            f[_lrx*8:_lrx*8+8,(_uly+1)*8:_lry*8] = np.dot(M,f[_lrx*8:_lrx*8+8,(_uly+1)*8:_lry*8])

        # 中
        f[_ulx*8+8:_lrx*8,_uly*8+8:_lry*8] = 0
    # 縦
    elif _ulx != _lrx:
        # 上
        f[_ulx*8:_ulx*8+8,_uly*8:_uly*8+8] = idct2(f[_ulx*8:_ulx*8+8,_uly*8:_uly*8+8])
        f[ulx:_ulx*8+8,uly:lry] = 0
        f[_ulx*8:_ulx*8+8,_uly*8:_uly*8+8] = dct2(f[_ulx*8:_ulx*8+8,_uly*8:_uly*8+8])
        # 下
        f[_lrx*8:_lrx*8+8,_uly*8:_uly*8+8] = idct2(f[_lrx*8:_lrx*8+8,_uly*8:_uly*8+8])
        f[_lrx*8:lrx,uly:lry] = 0
        f[_lrx*8:_lrx*8+8,_uly*8:_uly*8+8] = dct2(f[_lrx*8:_lrx*8+8,_uly*8:_uly*8+8])

        M = np.zeros((8,8))
        for i in range(uly%8):
            M[i,i] = 1
        if lry%8 != 0:
            for i in range(lry%8,8):
                M[i,i] = 1
        M = dct2(M)
        f[(_ulx+1)*8:_lrx*8,_lry*8:_lry*8+8] = np.dot(f[(_ulx+1)*8:_lrx*8,_lry*8:_lry*8+8],M)

    # 横
    elif _uly != _lry:
        # 左
        f[_ulx*8:_ulx*8+8,_uly*8:_uly*8+8] = idct2(f[_ulx*8:_ulx*8+8,_uly*8:_uly*8+8])
        f[ulx:lrx,uly:_uly*8+8] = 0
        f[_ulx*8:_ulx*8+8,_uly*8:_uly*8+8] = dct2(f[_ulx*8:_ulx*8+8,_uly*8:_uly*8+8])
        # 右
        f[_ulx*8:_ulx*8+8,_lry*8:_lry*8+8] = idct2(f[_ulx*8:_ulx*8+8,_lry*8:_lry*8+8])
        f[ulx:lrx,_lry*8:lry] = 0
        f[_ulx*8:_ulx*8+8,_lry*8:_lry*8+8] = dct2(f[_ulx*8:_ulx*8+8,_lry*8:_lry*8+8])

        M = np.zeros((8,8))
        for i in range(ulx%8):
            M[i,i] = 1
        if lrx%8 != 0:
            for i in range(lrx%8,8):
                M[i,i] = 1
        M = dct2(M)
        f[_lrx*8:_lrx*8+8,(_uly+1)*8:_lry*8] = np.dot(M,f[_lrx*8:_lrx*8+8,(_uly+1)*8:_lry*8])
    else:
        f[_ulx*8:_ulx*8+8,_uly*8:_uly*8+8] = idct2(f[_ulx*8:_ulx*8+8,_uly*8:_uly*8+8])
        f[ulx:lrx,uly:lry] = 0
        f[_ulx*8:_ulx*8+8,_uly*8:_uly*8+8] = dct2(f[_ulx*8:_ulx*8+8,_uly*8:_uly*8+8])
    return f