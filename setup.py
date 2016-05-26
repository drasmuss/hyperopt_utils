import io

from setuptools import find_packages, setup


def read(*filenames, **kwargs):
    encoding = kwargs.get('encoding', 'utf-8')
    sep = kwargs.get('sep', '\n')
    buf = []
    for filename in filenames:
        with io.open(filename, encoding=encoding) as f:
            buf.append(f.read())
    return sep.join(buf)



description = "hyperopt utils"
long_description = read('README.rst')

setup(
    name="hyperopt_utils",
    version="0.1.0",
    author="Daniel Rasmussen",
    author_email="daniel.rasmussen@appliedbrainresearch.com",
    packages=find_packages(),
    include_package_data=True,
    scripts=[],
    url="https://github.com/drasmuss/hyperopt_utils",
    license="",
    description=description,
    long_description=long_description,
    install_requires=["numpy", "hyperopt", "matplotlib"],
)
