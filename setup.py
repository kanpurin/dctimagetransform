from setuptools import setup, find_packages

setup(
    name="dctimagetransform",
    version='1.0.0',
    description='DCT係数上での高速な画像変換',
    author='Kanchi Hibi',
    author_email='kanpurin2@gmail.com',
    install_requires=['numpy'],
    url='https://github.com/kanpurin/dctimagetransform',
    packages=find_packages("dct_image_transform"),
    package_dir={"": "dct_image_transform"},
    test_suite='tests',
    include_package_data=True,
    license='MIT'
)