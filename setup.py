import io
import os
import re
import setuptools

with open('simutools/__init__.py') as fd:
    __version__ = re.search("__version__ = '(.*)'", fd.read()).group(1)


def read(*filenames, **kwargs):
    encoding = kwargs.get('encoding', 'utf-8')
    sep = kwargs.get('sep', '\n')
    buf = []
    for filename in filenames:
        with io.open(filename, encoding=encoding) as f:
            buf.append(f.read())
    return sep.join(buf)


long_description = read('README.md')

setuptools.setup(
    name='simutools',
    version=__version__,
    python_requires='>=3.8',
    install_requires=[
        'typed-argument-parser',
        'rdkit',
        'mdanalysis',
    ],
    author='Yan Xiang',
    author_email='1993.xiangyan@gmail.com',
    description='Tools for molecular simulation.',
    long_description=long_description,
    url='https://github.com/xiangyan93/simutools',
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    include_package_data=True,
    data_files=[('templates', [f'simutools/template/amber/{file}' for file in os.listdir('simutools/template/amber') if not file.endswith('py')]
                 + [f'simutools/template/gromacs/{file}' for file in os.listdir('simutools/template/gromacs') if not file.endswith('py')]
                 + [f'simutools/template/plumed/{file}' for file in os.listdir('simutools/template/plumed') if not file.endswith('py')])]
)
