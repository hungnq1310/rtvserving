from setuptools import setup, find_packages

setup(
    name="rtvserving",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        # "fastapi[all]",
        # "uvicorn",
        # "python-multipart",
        # "trism",
        "pydantic_settings"
    ],
    entry_points={
        "console_scripts": [
            # Add any CLI commands here if needed
        ],
    },
)