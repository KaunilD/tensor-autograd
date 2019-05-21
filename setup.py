import re
import os
from setuptools import setup

version = '1.0.0'
description = ''

with open('README.md', 'rb') as file:
    description = file.read().decode('utf-8')

setup(
    name='tensor_autograd',
    version=version,
    description='A pedagogical implementation of Automatic Differation on multi-dimensional tensors.',
    long_description=description,
    entry_points = {
        'console_scripts': ['tensor_autograd = tensor_autograd.tensor_autograd:main']
    },

    url='http://kaunild.github.io',
    author='Kaunil Dhruv',
    author_email='dhruv.kaunil@gmail.com',
    license='BSD',
    packages=['tensor_autograd', 'test']
)
