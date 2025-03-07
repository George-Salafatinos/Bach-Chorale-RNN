from setuptools import setup, find_packages

# Read requirements
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="bach_chorale_generator",
    version="0.1.0",
    description="A deep learning model to generate Bach-style chorales",
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    python_requires=">=3.10",
)