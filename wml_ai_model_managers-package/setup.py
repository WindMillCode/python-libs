from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'A conversion package'
LONG_DESCRIPTION = 'A package that makes it easy to convert values between several units of measurement'

setup(
    name="wml_ai_model_managers",
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    author="Windmillcode",
    author_email="dev@windmillcode.com",
    license='MIT',
    packages=find_packages(),
    install_requires=[],
    keywords=['ai','ml','ml train','ml test','pytorch','ml text'],
    classifiers= [
        "Intended Audience :: Developers",
        'License :: OSI Approved :: MIT License',
        "Programming Language :: Python :: 3",
    ]
)
