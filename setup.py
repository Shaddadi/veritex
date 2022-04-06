from setuptools import find_packages, setup
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='veritx',
    version='0.0.0',
    description='Tool for Reachability Analysis and Repair of Neural Networks.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/Shaddadi/veritex',
    author='Xiaodong Yang',
    author_email='xiaodong.yang@vanderbilt.edu',
    license='BSD',
    python_requires='>=3.7.0, <3.8',
    install_requires=[
        'torch==1.10.0',
        'numpy==1.21.4',
        'scipy==1.7.2',
        'onnx==1.10.2',
        'onnxruntime==1.9.0',
        'onnx2pytorch==0.4.1',
        'diffabs==0.1',
        'matplotlib',
        'future'
    ],
    packages=find_packages(),

    classifiers=[
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3.7',
    ],
)
