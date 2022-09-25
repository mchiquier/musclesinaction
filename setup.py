from setuptools import setup
from importlib.machinery import SourceFileLoader

with open('README.md') as file:
    long_description = file.read()

name='template'
version = SourceFileLoader(name + '.version', name + '/version.py').load_module()

setup(
   name=name,
   version=version.version,
   description='<Enter short description here>',
   author='<Enter your name(s) here>',
   author_email='<Enter your email(s) here>',
   url='TBD',
   packages=[name],
   long_description=long_description,
   long_description_content_type='text/markdown',
   keywords='<Enter keywords here separated by spaces>',
   license='',
   install_requires=[],
)