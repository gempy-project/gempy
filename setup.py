from setuptools import setup, find_packages

version = '2023.1.0b3'


def read_requirements(file_name):
    with open(file_name, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='gempy',
    version=version,
    packages=find_packages(exclude=('test', 'docs', 'examples')),
    install_requires=read_requirements("requirements.txt"),
    extras_require={
        "dev": read_requirements("dev-requirements.txt"),
        "opt": read_requirements("optional-requirements.txt"),
    },
    url='https://github.com/cgre-aachen/gempy',
    license='EUPL-1.2',
    author='Miguel de la Varga, Alexander Zimmerman, Elisa Heim, Alexander Schaaf, Fabian Stamm, Florian Wellmann, Jan Niederau, Andrew Annex',
    author_email='miguel@terranigma-solutions.com',
    description='An Open-source, Python-based 3-D structural geological modeling software.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords=['geology', '3-D modeling', 'structural geology', 'uncertainty'],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: European Union Public Licence 1.2 (EUPL 1.2)",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
    ],
)
