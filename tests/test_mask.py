import unittest
from skimage import io
import numpy as np
from dct_image_transform.dct import dct2, idct2
from dct_image_transform.mask import mask
import skimage.util
import random

class TestMaskVertical(unittest.TestCase):
    def test_mask_vertical_1(self):
        image = np.array(io.imread('images/sample512x512.jpg'),dtype=np.float)
        ulx, lrx, uly, lry = 203, 304, 105, 108
        tf_image = idct2(mask(dct2(image),ulx=ulx,lrx=lrx,uly=uly,lry=lry))
        image[ulx:lrx,uly:lry] = 0
        self.assertTrue(np.max(np.abs(image-tf_image)) < 1e-10)
        
    def test_mask_vertical_2(self):
        image = np.array(io.imread('images/sample1024x1024.jpg'),dtype=np.float)
        ulx, lrx, uly, lry = 200, 304, 105, 108
        tf_image = idct2(mask(dct2(image),ulx=ulx,lrx=lrx,uly=uly,lry=lry))
        image[ulx:lrx,uly:lry] = 0
        self.assertTrue(np.max(np.abs(image-tf_image)) < 1e-10)
        
    def test_mask_vertical_3(self):
        image = np.array(io.imread('images/sample2048x2048.jpg'),dtype=np.float)
        ulx, lrx, uly, lry = 200, 300, 104, 112
        tf_image = idct2(mask(dct2(image),ulx=ulx,lrx=lrx,uly=uly,lry=lry))
        image[ulx:lrx,uly:lry] = 0
        self.assertTrue(np.max(np.abs(image-tf_image)) < 1e-10)
        
    def test_mask_vertical_4(self):
        image = np.array(io.imread('images/sample512x512.jpg'),dtype=np.float)
        ulx, lrx, uly, lry = 200, 304, 104, 112
        tf_image = idct2(mask(dct2(image),ulx=ulx,lrx=lrx,uly=uly,lry=lry))
        image[ulx:lrx,uly:lry] = 0
        self.assertTrue(np.max(np.abs(image-tf_image)) < 1e-10)

class TestMaskHorizontal(unittest.TestCase):
    def test_mask_horizontal_1(self):
        image = np.array(io.imread('images/sample512x512.jpg'),dtype=np.float)
        ulx, lrx, uly, lry = 105, 108, 203, 304
        tf_image = idct2(mask(dct2(image),ulx=ulx,lrx=lrx,uly=uly,lry=lry))
        image[ulx:lrx,uly:lry] = 0
        self.assertTrue(np.max(np.abs(image-tf_image)) < 1e-10)
        
    def test_mask_horizontal_2(self):
        image = np.array(io.imread('images/sample1024x1024.jpg'),dtype=np.float)
        ulx, lrx, uly, lry = 105, 108, 200, 304
        tf_image = idct2(mask(dct2(image),ulx=ulx,lrx=lrx,uly=uly,lry=lry))
        image[ulx:lrx,uly:lry] = 0
        self.assertTrue(np.max(np.abs(image-tf_image)) < 1e-10)
        
    def test_mask_horizontal_3(self):
        image = np.array(io.imread('images/sample2048x2048.jpg'),dtype=np.float)
        ulx, lrx, uly, lry = 104, 112, 200, 300
        tf_image = idct2(mask(dct2(image),ulx=ulx,lrx=lrx,uly=uly,lry=lry))
        image[ulx:lrx,uly:lry] = 0
        self.assertTrue(np.max(np.abs(image-tf_image)) < 1e-10)
        
    def test_mask_horizontal_4(self):
        image = np.array(io.imread('images/sample512x512.jpg'),dtype=np.float)
        ulx, lrx, uly, lry = 104, 112, 200, 304
        tf_image = idct2(mask(dct2(image),ulx=ulx,lrx=lrx,uly=uly,lry=lry))
        image[ulx:lrx,uly:lry] = 0
        self.assertTrue(np.max(np.abs(image-tf_image)) < 1e-10)

class TestMaskSmall(unittest.TestCase):
    def test_mask_small_1(self):
        image = np.array(io.imread('images/sample512x512.jpg'),dtype=np.float)
        ulx, lrx, uly, lry = 105, 108, 104, 112
        tf_image = idct2(mask(dct2(image),ulx=ulx,lrx=lrx,uly=uly,lry=lry))
        image[ulx:lrx,uly:lry] = 0
        self.assertTrue(np.max(np.abs(image-tf_image)) < 1e-10)
        
    def test_mask_small_2(self):
        image = np.array(io.imread('images/sample1024x1024.jpg'),dtype=np.float)
        ulx, lrx, uly, lry = 104, 112, 105, 108
        tf_image = idct2(mask(dct2(image),ulx=ulx,lrx=lrx,uly=uly,lry=lry))
        image[ulx:lrx,uly:lry] = 0
        self.assertTrue(np.max(np.abs(image-tf_image)) < 1e-10)
        
    def test_mask_small_3(self):
        image = np.array(io.imread('images/sample2048x2048.jpg'),dtype=np.float)
        ulx, lrx, uly, lry = 104, 112, 104, 112
        tf_image = idct2(mask(dct2(image),ulx=ulx,lrx=lrx,uly=uly,lry=lry))
        image[ulx:lrx,uly:lry] = 0
        self.assertTrue(np.max(np.abs(image-tf_image)) < 1e-10)
        
    def test_mask_small_4(self):
        image = np.array(io.imread('images/sample512x512.jpg'),dtype=np.float)
        ulx, lrx, uly, lry = 105, 108, 105, 108
        tf_image = idct2(mask(dct2(image),ulx=ulx,lrx=lrx,uly=uly,lry=lry))
        image[ulx:lrx,uly:lry] = 0
        self.assertTrue(np.max(np.abs(image-tf_image)) < 1e-10)

class TestMaskLarge(unittest.TestCase):
    def test_mask_large_1(self):
        image = np.array(io.imread('images/sample512x512.jpg'),dtype=np.float)
        ulx, lrx, uly, lry = 203, 304, 203, 304
        tf_image = idct2(mask(dct2(image),ulx=ulx,lrx=lrx,uly=uly,lry=lry))
        image[ulx:lrx,uly:lry] = 0
        self.assertTrue(np.max(np.abs(image-tf_image)) < 1e-10)
        
    def test_mask_large_2(self):
        image = np.array(io.imread('images/sample1024x1024.jpg'),dtype=np.float)
        ulx, lrx, uly, lry = 203, 304, 200, 400
        tf_image = idct2(mask(dct2(image),ulx=ulx,lrx=lrx,uly=uly,lry=lry))
        image[ulx:lrx,uly:lry] = 0
        self.assertTrue(np.max(np.abs(image-tf_image)) < 1e-10)
        
    def test_mask_large_3(self):
        image = np.array(io.imread('images/sample2048x2048.jpg'),dtype=np.float)
        ulx, lrx, uly, lry = 200, 400, 203, 304
        tf_image = idct2(mask(dct2(image),ulx=ulx,lrx=lrx,uly=uly,lry=lry))
        image[ulx:lrx,uly:lry] = 0
        self.assertTrue(np.max(np.abs(image-tf_image)) < 1e-10)
        
    def test_mask_large_4(self):
        image = np.array(io.imread('images/sample512x512.jpg'),dtype=np.float)
        ulx, lrx, uly, lry = 200, 400, 200, 400
        tf_image = idct2(mask(dct2(image),ulx=ulx,lrx=lrx,uly=uly,lry=lry))
        image[ulx:lrx,uly:lry] = 0
        self.assertTrue(np.max(np.abs(image-tf_image)) < 1e-10)

class TestMaskRandom(unittest.TestCase):
    def test_mask_small_1(self):
        image = np.array(io.imread('images/sample512x512.jpg'),dtype=np.float)
        for _ in range(100):
            original_image = image.copy()
            ulx = random.randint(0,image.shape[0]-1)
            uly = random.randint(0,image.shape[1]-1)
            lrx = random.randint(ulx+1,image.shape[0])
            lry = random.randint(uly+1,image.shape[1])
            tf_image = idct2(mask(dct2(image),ulx=ulx,lrx=lrx,uly=uly,lry=lry))
            original_image[ulx:lrx,uly:lry] = 0
            self.assertTrue(np.max(np.abs(original_image-tf_image)) < 1e-10)

if __name__ == '__main__':
    unittest.main()