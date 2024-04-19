# -*- coding: utf-8 -*-

# Learn more: https://github.com/kennethreitz/setup.py

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='dl-with-cifar10',
    version='1.0.0',
    description='Deep Learning with Cifar-10 Datasets',
    long_description=readme,
    author='Patrick Zhang',
    author_email='zhanghaopai@outlook.com',
    url='https://github.com/zhanghaopai/ucas-deeplearning-project.git',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)

