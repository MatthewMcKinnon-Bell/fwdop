from setuptools import setup, find_packages

setup(
    name='fwdop',
    version='0.0.1',
    description='Forward operator helper for SensRay/pygeoinf integration',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'pygeoinf>=1.3.1'
    ],
    include_package_data=True,
    author='auto-generated',
)
