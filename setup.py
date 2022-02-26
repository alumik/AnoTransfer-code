import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='anomalytransfer',
    version='0.3.0',
    author='AlumiK',
    author_email='nczzy1997@gmail.com',
    license='MIT',
    description='Implementation of AnomalyTransfer in PyTorch',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/AlumiK/anomalytransfer',
    packages=setuptools.find_packages(include=['anomalytransfer', 'anomalytransfer.*']),
    platforms='any',
    install_requires=[
        'pandas',
        'scikit-learn',
        'torch',
        'tqdm'
    ],
    extras_require={
        'dev': [
            'matplotlib',
        ],
    },
    dependency_links=[
        'https://download.pytorch.org/whl/torch_stable.html',
    ],
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    python_requires='==3.8',
)
