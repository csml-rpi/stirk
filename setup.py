from setuptools import setup, find_packages

setup(
    name="STIRK",  # Replace with your package name
    version="0.1.0",  # Initial version of your package
    author="Shahriar Akbar Sakib",  # Replace with your name
    author_email="sakibs@rpi.edu",  # Replace with your email
    description="We propose a novel learning framework for Koopman operator of nonlinear dynamical systems that is informed by the governing equation and guarantees long-time stability and robustness to noise. In contrast to existing frameworks where either ad-hoc observables or blackbox neural networks are used to construct observables in the extended dynamic mode decomposition (EDMD), our observables are informed by governing equations via Polyflow. To improve the noise robustness and guarantee long-term stability, we designed a stable parameterization of the Koopman operator together with a progressive learning strategy for roll-out recurrent loss. To further improve model performance in the phase space, a simple iterative strategy of data augmentation was developed. Numerical experiments of prediction and control of classic nonlinear systems with ablation study showed the effectiveness of the proposed techniques over several state-of-the-art practices.",  # Short description
    long_description=open("README.md").read(),  # Detailed description from a README file
    long_description_content_type="text/markdown",  # Type of the README file
    url="https://github.com/orgs/csml-rpi/stirk",  # URL of the project's homepage or repository
    packages=find_packages(),  # Automatically find packages in your directory
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # Replace with your license if different
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",  # Minimum Python version required
    install_requires=[
        "numpy==1.24.4",
        "pandas==2.0.3",
        "scipy==1.10.1",
        "do-mpc==4.6.4",
        "matplotlib==3.7.4",
        "SciencePlots==2.1.1",
        "seaborn==0.13.2",
        "torch==2.1.1",
        "torchdiffeq==0.2.3"
    ],
)
