from setuptools import find_packages, setup

from torchprofile import __version__

setup(
    name='torchprofile',
    version=__version__,
    packages=find_packages(exclude=['examples']),
    install_requires=[
        'numpy>=1.14',
    ],
    url='https://github.com/zhijian-liu/torchprofile/',
    license='MIT',
)
