from setuptools import setup
from setuptools import find_namespace_packages

setup(
    name='asreview-XGBoost',
    version='0.1',
    description='A XGBoost Model for ASReview',
    url='https://github.com/asreview/asreview',
    author='Jelle Teijema',
    author_email='asreview@uu.nl',
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    keywords='systematic review',
    packages=find_namespace_packages(include=['asreviewcontrib.*']),
    python_requires='~=3.6',
    install_requires=[
        'sklearn',
        'asreview>=0.18',
        'xgboost'
    ],
    entry_points={
        'asreview.models.classifiers': [
            'xgboost = asreviewcontrib.models.xgboost:XGBoost',
        ],
        'asreview.models.feature_extraction': [
            # define feature_extraction algorithms
        ],
        'asreview.models.balance': [
            # define balance strategy algorithms
        ],
        'asreview.models.query': [
            # define query strategy algorithms
        ]
    },
    project_urls={
        'Bug Reports': 'https://github.com/JTeijema/asreview-XGBoost/issues',
        'Source': 'https://github.com/JTeijema/asreview-XGBoost',
    },
)
