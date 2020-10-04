import setuptools

setuptools.setup(
    name="mgdl",
    version="0.0.1",
    author="Maochen",
    author_email="contact@maochen.org",
    description="MG DL Toolkit",
    long_description="",
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],

    install_requires=[
        "matplotlib>=3.1.3",
        "numpy",
        "scikit-plot>=0.3.7",
        "sklearn",
        "tqdm",
        "torch",
        "torchvision",
        "tensorboardX",
        "transformers"
    ],
    dependency_links=[
    ],

    extras_require={

    }
)
