from setuptools import setup, find_packages, Extension

version = '2.3.0'

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='gempy',
    version=version,
    packages=find_packages(exclude=('test', 'docs', 'examples')),
    include_package_data=True,
    install_requires=[
        'pandas==2.0.1',
        'aesara',
        'pymc',
        'matplotlib',
        'numpy',
        'pytest',
        'seaborn>=0.9',
        'networkx',
        'scikit-image>=0.17',
        'scikit-learn',
        'pyvista==0.39.1',
        'pyvistaqt==0.10.0',
        'pyqt5',
        'iPython',
        'rasterio',
        'xarray'
    ],
    url='https://github.com/cgre-aachen/gempy',
    license='LGPL v3',
    author='Miguel de la Varga, Alexander Zimmerman, Elisa Heim, Alexander Schaaf, Fabian Stamm, Florian Wellmann, Jan Niederau, Andrew Annex, Alexander Juestel',
    author_email='miguel@terranigma-solutions.com',
    description='An Open-source, Python-based 3-D structural geological modeling software.',
    long_description='GemPy is a Python-based, open-source geomodeling library. It is capable of constructing complex 3D geological models of folded structures, fault networks and unconformities, based on the underlying powerful implicit representation approach.' ,
    keywords=['geology', '3-D modeling', 'structural geology', 'uncertainty']
)
