import setuptools

setuptools.setup(packages=setuptools.find_packages(exclude=('myExample')),
                 include_package_data=True,
                 install_requires=['numpy', 'opencv-python', 'tifffile', 'pydicom', 'natsort'])