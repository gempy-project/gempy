from os import path

from setuptools import setup, find_packages

def read_requirements(file_name):
    requirements = []
    with open(file_name, "r", encoding="utf-8") as f:
        for line in f:
            # Strip whitespace and ignore comments
            line = line.strip()
            if line.startswith("#") or not line:
                continue

            # Handle -r directive
            if line.startswith("-r "):
                referenced_file = line.split()[1]  # Extract the file name
                requirements.extend(read_requirements(referenced_file))  # Recursively read referenced file
            else:
                requirements.append(line)

    return requirements


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='gempy',
    packages=find_packages(exclude=('test', 'docs', 'examples')),
    install_requires=read_requirements("requirements/requirements.txt"),
    extras_require={
        "opt": read_requirements("requirements/optional-requirements.txt"),
        "base": read_requirements("requirements/base-requirements.txt"),
    },
    url='https://github.com/cgre-aachen/gempy',
    license='EUPL-1.2',
    author='Miguel de la Varga, Alexander Zimmerman, Elisa Heim, Alexander Schaaf, Fabian Stamm, Florian Wellmann, Jan Niederau, Andrew Annex',
    author_email='gempy@terranigma-solutions.com',
    description='An Open-source, Python-based 3-D structural geological modeling software.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords=['geology', '3-D modeling', 'structural geology', 'uncertainty'],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: European Union Public Licence 1.2 (EUPL 1.2)",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    setup_requires=['setuptools_scm'],
    use_scm_version={
            "root"            : ".",
            "relative_to"     : __file__,
            "write_to"        : path.join("gempy", "_version.py"),
            "fallback_version": "3.0.0"
    },
)
