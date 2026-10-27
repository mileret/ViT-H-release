from setuptools import find_namespace_packages, setup


setup(
    name="vith-offline",
    version="1.0.0",
    packages=find_namespace_packages(include=("hmr4d*", "motion_lib*")),
    py_modules=["canonicalize_motion", "roma"],
    include_package_data=True,
    description="ViT-H G1 motion training and offline inference",
    python_requires=">=3.10",
)
