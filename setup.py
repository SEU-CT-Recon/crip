import setuptools

setuptools.setup(packages=setuptools.find_packages(),
                 include_package_data=True,
                 install_requires=['numpy', 'opencv-python', 'tifffile', 'pydicom', 'natsort'])