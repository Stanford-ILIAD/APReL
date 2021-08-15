from setuptools import setup, find_packages

# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.rst'), encoding='utf-8') as f:
    lines = f.readlines()

# remove images from README
lines = [x for x in lines if '.png' not in x and '.gif' not in x]
long_description = ''.join(lines)

setup(
    name="aprel",
    packages=[
        package for package in find_packages() if package.startswith("aprel")
    ],
    install_requires=[
        "numpy>=1.8.0",
        "moviepy>=1.0.0",
        "scipy",
        "gym",
        "pygame"
    ],
    eager_resources=['*'],
    include_package_data=True,
    python_requires='>=3',
    description="APReL: APReL: A Library for Active Preference-based Reward Learning Algorithms",
    author="Erdem Biyik, Aditi Talati, Dorsa Sadigh",
    url="https://github.com/Stanford-ILIAD/aprel",
    author_email="ebiyik@stanford.edu",
    version="1.0.0",
    long_description=long_description,
    long_description_content_type='text/markdown'
)
