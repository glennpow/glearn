import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="learning",
    version="0.0.1",
    author="Glenn Powell",
    author_email="glenn@openai.com",
    description="Personal learning projects",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/glennpow/learning",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
