"""Setup script."""
from setuptools import setup

setup(name='claimit',
      version='0.1',
      description='A framework from detecting and clustering claims in online discussions.',
      url='https://github.com/avouacr/claimit',
      author='Romain Avouac',
      author_email='avouacr@gmail.com',
      license='MIT',
      packages=['claimit'],
      zip_safe=False,
      install_requires=[
          'joblib',
          'numpy',
          'pandas',
          'scikit-learn',
          'torch',
          'transformers',
          'umap-learn',
          'hdbscan',
          ],
      extras_require={
          'visualization': [
              'bokeh',
              'matplotlib',
              'datashader',
              'holoviews'
              ]
          }
      )
