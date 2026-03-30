from setuptools import setup, find_packages

# Load the version info.
#
# Note that we cannot simply import the module, since dependencies listed
# in setup() will very likely not be installed yet when setup.py run.
#
# See:
#   https://packaging.python.org/guides/single-sourcing-package-version

__version__ = None

with open('src/sen_et_openeo/_version.py') as fp:
    exec(fp.read())
    version = __version__


# Configure setuptools

setup(
    name="sen-et-openeo",
    version=version,
    author="Jeroen Degerickx, Astrid Vannoppen",
    author_email="jeroen.degerickx@vito.be, astrid.vannoppen@vito.be",
    description=(
        "SenET OpenEO workflow for computing LST-Ta (Land Surface Temperature "
        "minus air temperature), LSTM-like 30 m LST, and evapotranspiration "
        "(TSEB-PT model) from Sentinel-2, Sentinel-3 and ERA5 data via OpenEO."
    ),
    url='',
    license="Property of VITO NV",
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    package_data={
        '': ['resources/*', '*.geojson'],
        'sen_et_openeo.layers': ['*.geojson'],
        'sen_et_openeo.utils': ['*.geojson'],
    },
    zip_safe=True,
    python_requires='>=3.11',
    install_requires=[
    ],
    test_suite='tests',
    package_dir={'': 'src'},
    packages=find_packages('src')
)
