import unittest
from skimage import io
import numpy as np
from dct_image_transform.dct import dct2, idct2
from dct_image_transform.reflection import reflection
import skimage.util

class TestReflection0(unittest.TestCase):
    def test_reflecion0_1(self):
        image = np.array(io.imread('images/sample512x512.jpg'),dtype=np.float)
        tf_image = idct2(reflection(dct2(image),axis=0))
        image = np.flipud(image)
        self.assertTrue(np.max(np.abs(image-tf_image)) < 1e-10)

    def test_reflecion0_2(self):
        image = np.array(io.imread('images/sample1024x1024.jpg'),dtype=np.float)
        tf_image = idct2(reflection(dct2(image),axis=0))
        image = np.flipud(image)
        self.assertTrue(np.max(np.abs(image-tf_image)) < 1e-10)
        
    def test_reflecion0_3(self):
        image = np.array(io.imread('images/sample2048x2048.jpg'),dtype=np.float)
        tf_image = idct2(reflection(dct2(image),axis=0))
        image = np.flipud(image)
        self.assertTrue(np.max(np.abs(image-tf_image)) < 1e-10)

class TestReflection1(unittest.TestCase):
    def test_reflecion1_1(self):
        image = np.array(io.imread('images/sample512x512.jpg'),dtype=np.float)
        tf_image = idct2(reflection(dct2(image),axis=1))
        image = np.fliplr(image)
        self.assertTrue(np.max(np.abs(image-tf_image)) < 1e-10)

    def test_reflecion1_2(self):
        image = np.array(io.imread('images/sample1024x1024.jpg'),dtype=np.float)
        tf_image = idct2(reflection(dct2(image),axis=1))
        image = np.fliplr(image)
        self.assertTrue(np.max(np.abs(image-tf_image)) < 1e-10)
        
    def test_reflecion1_3(self):
        image = np.array(io.imread('images/sample2048x2048.jpg'),dtype=np.float)
        tf_image = idct2(reflection(dct2(image),axis=1))
        image = np.fliplr(image)
        self.assertTrue(np.max(np.abs(image-tf_image)) < 1e-10)

if __name__ == '__main__':
    unittest.main()