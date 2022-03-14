import setuptools

setuptools.setup(name="crip",
                 version="0.4",
                 author="CandleHouse",
                 author_email="zx.cs@qq.coom",
                 description="Cone-Beam CT Data IO, Pre/Post-process and Physics",
                 url="https://github.com/z0gSh1u/crip",
                 packages=setuptools.find_packages(exclude=('myExample')),
                 install_requires=['numpy', 'opencv-python', 'tifffile', 'pydicom', 'natsort'])