from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
  long_description = fh.read()

setup(
    name='raa',
    version='0.1',
    # packages=['raa'],
    packages=find_packages(),
    install_requires=[
        # Add your project dependencies here
    ],
    entry_points={
        'console_scripts': [
            'raa=raa.main:main'
        ]
    },
    author='Kelvin Chan',
    author_email='kechan.ca@gmail.com',
    description='Retrieval Augmented Agent',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/kechan/raa',
    license=None,
)
