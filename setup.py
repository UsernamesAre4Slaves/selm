from setuptools import setup, find_packages

setup(
    name='SELM',
    version='0.1.0',
    description='Small Efficient Language Model (SELM)',
    author='[Your Name]',
    author_email='[Your Email]',
    url='https://github.com/[YourUsername]/SELM',  # Replace with your repository URL
    packages=find_packages(),
    install_requires=[
        'torch>=1.9.0',
        'transformers>=4.0.0',
        'datasets>=1.0.0',
        'optuna>=2.0.0',
        'torch_pruning>=0.1.0',
        'scikit-learn>=0.24.0',
        'tqdm>=4.0.0',
        'pyyaml>=5.4.0',
    ],
    entry_points={
        'console_scripts': [
            'train=src.scripts.train:main',
            'evaluate=src.scripts.evaluate:main',
            'prune_and_quantize=src.scripts.prune_and_quantize:main',
            'optuna_tuning=src.scripts.run_optuna_tuning:main',
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
