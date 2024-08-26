from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'My first Python  ML package'

# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="linear_regressions", 
        version=VERSION,
        author="Jamarcus Williams",
        author_email="jamarcuswiliams12@gmail.com",
        description=DESCRIPTION,
        packages=find_packages(),
        install_requires=[
            'caer',
            'seaborn',
            'pandas',
            'numpy',
            'plotly',
            'keras',
        ], # add any additional packages that 
        # needs to be installed along with your package. Eg: 'caer'
        
        keywords=['python', 'basic model', 'linear regression', 'machine learning'], # Keywords that define your package best
)