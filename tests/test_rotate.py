import unittest
from skimage import io
import numpy as np
from dct_image_transform.dct import dct2, idct2
from dct_image_transform.rotate import rotate90, rotate180, rotate270
import skimage.util

class TestRotate90(unittest.TestCase):
    def test_rotate90_1(self):
        image = np.array(io.imread('images/sample512x512.jpg'),dtype=np.float)
        tf_image = idct2(rotate90(dct2(image)))
        image = np.rot90(image,k=1)
        self.assertTrue(np.max(np.abs(image-tf_image)) < 1e-10)
        
    def test_rotate90_2(self):
        image = np.array(io.imread('images/sample1024x1024.jpg'),dtype=np.float)
        tf_image = idct2(rotate90(dct2(image)))
        image = np.rot90(image,k=1)
        self.assertTrue(np.max(np.abs(image-tf_image)) < 1e-10)
        
    def test_rotate90_3(self):
        image = np.array(io.imread('images/sample2048x2048.jpg'),dtype=np.float)
        tf_image = idct2(rotate90(dct2(image)))
        image = np.rot90(image,k=1)
        self.assertTrue(np.max(np.abs(image-tf_image)) < 1e-10)

class TestRotate180(unittest.TestCase):
    def test_rotate180_1(self):
        image = np.array(io.imread('images/sample512x512.jpg'),dtype=np.float)
        tf_image = idct2(rotate180(dct2(image)))
        image = np.rot90(image,k=2)
        self.assertTrue(np.max(np.abs(image-tf_image)) < 1e-10)
        
    def test_rotate180_2(self):
        image = np.array(io.imread('images/sample1024x1024.jpg'),dtype=np.float)
        tf_image = idct2(rotate180(dct2(image)))
        image = np.rot90(image,k=2)
        self.assertTrue(np.max(np.abs(image-tf_image)) < 1e-10)
        
    def test_rotate180_3(self):
        image = np.array(io.imread('images/sample2048x2048.jpg'),dtype=np.float)
        tf_image = idct2(rotate180(dct2(image)))
        image = np.rot90(image,k=2)
        self.assertTrue(np.max(np.abs(image-tf_image)) < 1e-10)
        
class TestRotate270(unittest.TestCase):
    def test_rotate270_1(self):
        image = np.array(io.imread('images/sample512x512.jpg'),dtype=np.float)
        tf_image = idct2(rotate270(dct2(image)))
        image = np.rot90(image,k=3)
        self.assertTrue(np.max(np.abs(image-tf_image)) < 1e-10)
        
    def test_rotate270_2(self):
        image = np.array(io.imread('images/sample1024x1024.jpg'),dtype=np.float)
        tf_image = idct2(rotate270(dct2(image)))
        image = np.rot90(image,k=3)
        self.assertTrue(np.max(np.abs(image-tf_image)) < 1e-10)
        
    def test_rotate270_3(self):
        image = np.array(io.imread('images/sample2048x2048.jpg'),dtype=np.float)
        tf_image = idct2(rotate270(dct2(image)))
        image = np.rot90(image,k=3)
        self.assertTrue(np.max(np.abs(image-tf_image)) < 1e-10)

if __name__ == '__main__':
    unittest.main()