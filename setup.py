from setuptools import setup, find_packages

setup(
    name="adaptive_game_ai",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'torch>=2.0.0',
        'gymnasium[atari]>=0.29.0',
        'numpy>=1.24.0',
        'matplotlib>=3.7.0',
        'opencv-python>=4.8.0'
    ]
) 