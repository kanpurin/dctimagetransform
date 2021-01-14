import unittest
from skimage import io
import numpy as np
from dct_image_transform.dct import dct2, idct2
from scipy.fftpack import dctn,idctn
import skimage.util

class TestDCT(unittest.TestCase):
    def test_dct2_1(self):
        image = np.array(io.imread('tests/images/sample512x512.jpg'),dtype=np.float)
        dct_image = dct2(image)
        blocks = skimage.util.view_as_blocks(image,(8,8))
        blocks[:] = dctn(blocks,axes=(2,3),norm='ortho')
        self.assertTrue(np.max(np.abs(image-dct_image)) < 1e-10)

    def test_dct2_2(self):
        image = np.array(io.imread('tests/images/sample1024x1024.jpg'),dtype=np.float)
        dct_image = dct2(image)
        blocks = skimage.util.view_as_blocks(image,(8,8))
        blocks[:] = dctn(blocks,axes=(2,3),norm='ortho')
        self.assertTrue(np.max(np.abs(image-dct_image)) < 1e-10)

    def test_dct2_3(self):
        image = np.array(io.imread('tests/images/sample2048x2048.jpg'),dtype=np.float)
        dct_image = dct2(image)
        blocks = skimage.util.view_as_blocks(image,(8,8))
        blocks[:] = dctn(blocks,axes=(2,3),norm='ortho')
        self.assertTrue(np.max(np.abs(image-dct_image)) < 1e-10)

class TestIDCT(unittest.TestCase):
    def test_idct2_1(self):
        image = np.array(io.imread('tests/images/sample512x512.jpg'),dtype=np.float)
        idct_image = idct2(image)
        blocks = skimage.util.view_as_blocks(image,(8,8))
        blocks[:] = idctn(blocks,axes=(2,3),norm='ortho')
        self.assertTrue(np.max(np.abs(image-idct_image)) < 1e-10)

    def test_idct2_2(self):
        image = np.array(io.imread('tests/images/sample1024x1024.jpg'),dtype=np.float)
        idct_image = idct2(image)
        blocks = skimage.util.view_as_blocks(image,(8,8))
        blocks[:] = idctn(blocks,axes=(2,3),norm='ortho')
        self.assertTrue(np.max(np.abs(image-idct_image)) < 1e-10)

    def test_idct2_3(self):
        image = np.array(io.imread('tests/images/sample2048x2048.jpg'),dtype=np.float)
        idct_image = idct2(image)
        blocks = skimage.util.view_as_blocks(image,(8,8))
        blocks[:] = idctn(blocks,axes=(2,3),norm='ortho')
        self.assertTrue(np.max(np.abs(image-idct_image)) < 1e-10)

class TestDCTIDCT(unittest.TestCase):
    def test_dctidct_1(self):
        image = np.array(io.imread('tests/images/sample512x512.jpg'),dtype=np.float)
        dct_image = idct2(dct2(image))
        self.assertTrue(np.max(np.abs(image-dct_image)) < 1e-10)

    def test_dctidct_2(self):
        image = np.array(io.imread('tests/images/sample1024x1024.jpg'),dtype=np.float)
        dct_image = idct2(dct2(image))
        self.assertTrue(np.max(np.abs(image-dct_image)) < 1e-10)

    def test_dctidct_3(self):
        image = np.array(io.imread('tests/images/sample2048x2048.jpg'),dtype=np.float)
        dct_image = idct2(dct2(image))
        self.assertTrue(np.max(np.abs(image-dct_image)) < 1e-10)

if __name__ == '__main__':
    unittest.main()