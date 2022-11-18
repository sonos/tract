from setuptools import setup
from setuptools_rust import Binding, RustExtension

setup(
    name="tract",
    version="1.0",
    rust_extensions=[RustExtension("tract.tract", binding=Binding.NoBinding, path="../Cargo.toml")],
    packages=["tract"],
    zip_safe=False,
)

