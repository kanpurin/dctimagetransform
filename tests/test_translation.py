import unittest
from skimage import io
import numpy as np
from dct_image_transform.dct import dct2, idct2
from dct_image_transform.translation import translation
import skimage.util

class TestTranslationPositive0(unittest.TestCase):
    def test_translationpositive0_1(self):
        image = np.array(io.imread('images/sample512x512.jpg'),dtype=np.float)
        tf_image = idct2(translation(dct2(image),100,axis=0))
        image[100:,:] = image[:-100,:]
        image[:100,:] = 0
        self.assertTrue(np.max(np.abs(image-tf_image)) < 1e-10)

    def test_translationpositive0_2(self):
        image = np.array(io.imread('images/sample1024x1024.jpg'),dtype=np.float)
        tf_image = idct2(translation(dct2(image),100,axis=0))
        image[100:,:] = image[:-100,:]
        image[:100,:] = 0
        self.assertTrue(np.max(np.abs(image-tf_image)) < 1e-10)
        
    def test_translationpositive0_3(self):
        image = np.array(io.imread('images/sample2048x2048.jpg'),dtype=np.float)
        tf_image = idct2(translation(dct2(image),100,axis=0))
        image[100:,:] = image[:-100,:]
        image[:100,:] = 0
        self.assertTrue(np.max(np.abs(image-tf_image)) < 1e-10)

class TestTranslationPositive1(unittest.TestCase):
    def test_translationpositive1_1(self):
        image = np.array(io.imread('images/sample512x512.jpg'),dtype=np.float)
        tf_image = idct2(translation(dct2(image),100,axis=1))
        image[:,100:] = image[:,:-100]
        image[:,:100] = 0
        self.assertTrue(np.max(np.abs(image-tf_image)) < 1e-10)

    def test_translationpositive1_2(self):
        image = np.array(io.imread('images/sample1024x1024.jpg'),dtype=np.float)
        tf_image = idct2(translation(dct2(image),100,axis=1))
        image[:,100:] = image[:,:-100]
        image[:,:100] = 0
        self.assertTrue(np.max(np.abs(image-tf_image)) < 1e-10)
        
    def test_translationpositive1_3(self):
        image = np.array(io.imread('images/sample2048x2048.jpg'),dtype=np.float)
        tf_image = idct2(translation(dct2(image),100,axis=1))
        image[:,100:] = image[:,:-100]
        image[:,:100] = 0
        self.assertTrue(np.max(np.abs(image-tf_image)) < 1e-10)

class TestTranslationNegative0(unittest.TestCase):
    def test_translationnegative0_1(self):
        image = np.array(io.imread('images/sample512x512.jpg'),dtype=np.float)
        tf_image = idct2(translation(dct2(image),-100,axis=0))
        image[:-100,:] = image[100:,:]
        image[-100:,:] = 0
        self.assertTrue(np.max(np.abs(image-tf_image)) < 1e-10)

    def test_translationnegative0_2(self):
        image = np.array(io.imread('images/sample1024x1024.jpg'),dtype=np.float)
        tf_image = idct2(translation(dct2(image),-100,axis=0))
        image[:-100,:] = image[100:,:]
        image[-100:,:] = 0
        self.assertTrue(np.max(np.abs(image-tf_image)) < 1e-10)
        
    def test_translationnegative0_3(self):
        image = np.array(io.imread('images/sample2048x2048.jpg'),dtype=np.float)
        tf_image = idct2(translation(dct2(image),-100,axis=0))
        image[:-100,:] = image[100:,:]
        image[-100:,:] = 0
        self.assertTrue(np.max(np.abs(image-tf_image)) < 1e-10)

class TestTranslationNegative1(unittest.TestCase):
    def test_translationnegative1_1(self):
        image = np.array(io.imread('images/sample512x512.jpg'),dtype=np.float)
        tf_image = idct2(translation(dct2(image),-100,axis=1))
        image[:,:-100] = image[:,100:]
        image[:,-100:] = 0
        self.assertTrue(np.max(np.abs(image-tf_image)) < 1e-10)

    def test_translationnegative1_2(self):
        image = np.array(io.imread('images/sample1024x1024.jpg'),dtype=np.float)
        tf_image = idct2(translation(dct2(image),-100,axis=1))
        image[:,:-100] = image[:,100:]
        image[:,-100:] = 0
        self.assertTrue(np.max(np.abs(image-tf_image)) < 1e-10)
        
    def test_translationnegative1_3(self):
        image = np.array(io.imread('images/sample2048x2048.jpg'),dtype=np.float)
        tf_image = idct2(translation(dct2(image),-100,axis=1))
        image[:,:-100] = image[:,100:]
        image[:,-100:] = 0
        self.assertTrue(np.max(np.abs(image-tf_image)) < 1e-10)   

class TestTranslationZero(unittest.TestCase):
    def test_translationzero_1(self):
        image = np.array(io.imread('images/sample512x512.jpg'),dtype=np.float)
        tf_image = idct2(translation(dct2(image),0,axis=0))

        self.assertTrue(np.max(np.abs(image-tf_image)) < 1e-10)

    def test_translationzero_2(self):
        image = np.array(io.imread('images/sample512x512.jpg'),dtype=np.float)
        tf_image = idct2(translation(dct2(image),0,axis=1))
        self.assertTrue(np.max(np.abs(image-tf_image)) < 1e-10)    
        
class TestTranslationZero(unittest.TestCase):
    def test_translationzero_1(self):
        image = np.array(io.imread('images/sample512x512.jpg'),dtype=np.float)
        tf_image = idct2(translation(dct2(image),0,axis=0))
        self.assertTrue(np.max(np.abs(image-tf_image)) < 1e-10)

    def test_translationzero_2(self):
        image = np.array(io.imread('images/sample512x512.jpg'),dtype=np.float)
        tf_image = idct2(translation(dct2(image),0,axis=1))
        self.assertTrue(np.max(np.abs(image-tf_image)) < 1e-10)  

class TestTranslationOver(unittest.TestCase):
    def test_translationzero_1(self):
        image = np.array(io.imread('images/sample512x512.jpg'),dtype=np.float)
        tf_image = idct2(translation(dct2(image),image.shape[0]*2,axis=0))
        image[:] = 0
        self.assertTrue(np.max(np.abs(image-tf_image)) < 1e-10)

    def test_translationzero_2(self):
        image = np.array(io.imread('images/sample512x512.jpg'),dtype=np.float)
        tf_image = idct2(translation(dct2(image),image.shape[1]*2,axis=0))
        image[:] = 0
        self.assertTrue(np.max(np.abs(image-tf_image)) < 1e-10) 

if __name__ == '__main__':
    unittest.main()