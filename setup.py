from setuptools import setup, find_packages

version = '2023.1.0'

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='gempy',
    version=version,
    packages=find_packages(exclude=('test', 'docs', 'examples')),
    include_package_data=True,
    install_requires=[
        'numpy'
    ],
    url='https://github.com/cgre-aachen/gempy',
    license='EUPL-1.2',
    author='Miguel de la Varga, Alexander Zimmerman, Elisa Heim, Alexander Schaaf, Fabian Stamm, Florian Wellmann, Jan Niederau, Andrew Annex',
    author_email='miguel@terranigma-solutions.com',
    description='An Open-source, Python-based 3-D structural geological modeling software.',
    long_description=long_description,
    keywords=['geology', '3-D modeling', 'structural geology', 'uncertainty']
)
