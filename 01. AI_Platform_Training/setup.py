from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
]

setup(
    name='trainer_rabbit',
    version='0.1',
    author = 'juan lopez',
    install_requires=["pandas"],
    packages=find_packages(),
    include_package_data=True,
    description='census model rabbit course',
    requires=[]
)