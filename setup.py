from setuptools import setup, find_packages

setup(
    name="chi_geometry",
    version="0.1.3",
    description="A library for generating and analyzing chiral-aware datasets and models.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Rylie Weaver",
    author_email="rylieweaver9@example.com",
    url="https://github.com/RylieWeaver/Chi-Geometry",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.5.0",
        "torch>=2.0.0",
        "torch_geometric>=2.0.0",
        "matplotlib>=3.6.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    project_urls={
        "Documentation": "https://github.com/RylieWeaver/Chi-Geometry",
        "Source": "https://github.com/RylieWeaver/Chi-Geometry",
        "Tracker": "https://github.com/RylieWeaver/Chi-Geometry/issues",
    },
)
