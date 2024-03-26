from setuptools import setup, find_packages

setup(
    name="simpipelinehki", 
    version="1.0", 
    author="Ruby J. Wright",  # Replace with your name
    author_email="ruby.wright@helsinki.fi",  # Replace with your email
    description="A package to analyse SMBH focused gadget simulations.",  # Replace with a brief description of your package
    long_description=open('README.md').read(),  # Make sure you have a README.md file
    long_description_content_type="text/markdown",
    url="http://github.com/yourusername/your-package",  # Replace with the URL of your package's source code
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)