import setuptools

setuptools.setup(packages=setuptools.find_packages(),
                 package_data={'': ['_atten/**/*.txt', '_atten/*.json']},
                 install_requires=['numpy', 'opencv-python', 'tifffile', 'pydicom', 'natsort'])