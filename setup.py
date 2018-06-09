from setuptools import setup, find_packages

setup(
    name='gempy',
    version='1.14',
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'theano',
        'scikit-image',
        'seaborn'
    ],
    url='https://github.com/cgre-aachen/gempy',
    download_url='https://github.com/cgre-aachen/gempy/archive/1.01.tar.gz',
    license='LGPL v3',
    author='Miguel de la Varga, Alexander Schaaf, Fabian Stamm, Florian Wellmann',
    author_email='varga@aices.rwth-aachen.de',
    description='An Open-source, Python-based 3-D structural geological modeling software.',
    keywords=['geology', '3-D modeling', 'structural geology', 'uncertainty']
)
