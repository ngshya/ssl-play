from setuptools import setup

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='sslplay',
    url='https://github.com/ngshya/ssl-play',
    author='ngshya',
    author_email='ngshya@gmail.com',
    packages=['sslplay', 'sslplay.data', 'sslplay.model', 'sslplay.performance', 'sslplay.utils'],
    install_requires=required,
    version='1.1.1',
    license=open("LICENSE").read(),
    description='Experiments with semi-supervised learning methods.',
    long_description=open('README.md').read(),
    package_data={
        'sslplay': ["*"],
        'notebooks': ['*'],
        'scripts': ['*']
    }
)