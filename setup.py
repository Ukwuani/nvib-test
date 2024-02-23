import setuptools

setuptools.setup(
    name="project-nvib",
    version="1",
    description="VAE for transformers with NVIB",
    url="/",
    author="UKB",
    install_requires=["torch"],
    packages=setuptools.find_packages(),
    zip_safe=False,
)