import setuptools

setuptools.setup(
    packages=setuptools.find_packages(exclude=['test', 'docs', 'script']),
    package_data={'': ['_asset/**/*']},
    install_requires=['numpy', 'opencv-python', 'tifffile', 'pydicom', 'natsort', 'matplotlib', 'scikit-image', 'periodictable'])
