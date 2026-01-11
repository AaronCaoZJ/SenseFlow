from setuptools import setup, find_packages

setup(
    name='senseflow',
    version='1.0',
    description='SenseFlow: Scaling Distribution Matching for Flow-based Text-to-Image Distillation',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'tqdm',
    ],
)