from setuptools import setup,find_packages
from typing import List

def get_req(file_path:str)->List[str]:
    require=[]
    with open(file_path) as file:
        require=file.readlines()
        require=[req.replace('\n','') for req in require]
        if '-e .' in require:
            require.remove('-e .')
        return require
setup(
    name='DiamondPricePrediction',
    version='0.0.1',
    author='Krishna',
    author_email='dragonkrishoredbzgt@gmail.com',
    install_requires=get_req('requirements.txt'),
    packages=find_packages()
)
