from setuptools import setup, find_packages

setup(name='sampling_based_lyapunov',
      version='1.0.0',
      packages=find_packages(),
      install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'casadi',
        'cvxpy',
        'mosek',
        'scikit-image'
        ]
      )
