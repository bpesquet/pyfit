"""Setup for the pyfit package."""

import setuptools


with open("README.md") as f:
    README = f.read()

setuptools.setup(
    author="Baptiste Pesquet",
    author_email="bpesquet@gmail.com",
    name="pyfit",
    license="MIT",
    description="A minimalist neural networks library written for educational purposes",
    version="0.1.1",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/bpesquet/pyfit",
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
    install_requires=[],
    classifiers=[
        # Trove classifiers
        # (https://pypi.python.org/pypi?%3Aaction=list_classifiers)
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Intended Audience :: Developers",
    ],
)
