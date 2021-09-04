import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="panter",
    version="0.0.3",
    author="Max Lamparth",
    author_email="max.lamparth@tum.de",
    description="panter - Perkeo ANalysis Tool for Evaluation and Reduction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/maxlampe/panter",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
