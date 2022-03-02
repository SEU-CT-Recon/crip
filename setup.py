import setuptools

setuptools.setup(packages=setuptools.find_packages(exclude=('myExample')),
                 install_requires=['numpy', 'opencv-python', 'tifffile', 'pydicom', 'natsort'])