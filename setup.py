from setuptools import setup, find_packages

setup(
    name="simpipeline_helsinki", 
    version="1.0", 
    author="Ruby J. Wright",  
    author_email="ruby.wright@helsinki.fi", 
    description="A python package to analyse SMBH focused GADGET simulations.", 
    long_description=open('README.md').read(),  # Make sure you have a README.md file
    long_description_content_type="text/markdown",
    url="https://github.com/RJWright25/simpipeline_helsinki",  
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=['numpy','pandas','matplotlib','scipy','astropy','h5py','py-sphviewer','moviepy']
)