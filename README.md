![CI](https://github.com/caponetto/bayesian-hierarchical-clustering/workflows/Python%20application/badge.svg)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://github.com/caponetto/bayesian-hierarchical-clustering/blob/main/LICENSE)

# Bayesian Hierarchical Clustering in python
This is a python implementation of the Bayesian Hierarchical Clustering algorithm proposed by Heller & Ghahramani (2005).
> HELLER, Katherine A.; GHAHRAMANI, Zoubin. Bayesian hierarchical clustering. In: **Proceedings of the 22nd international conference on Machine learning**. 2005. p. 297-304.

It also includes the Bayesian Rose Trees extension proposed by Blundell et al (2012).
> BLUNDELL, Charles; TEH, Yee Whye; HELLER, Katherine A. Bayesian rose trees. arXiv preprint arXiv:1203.3468, 2012.

## How to build from source
1. Create an anaconda environment using the [environment.yml](environment.yml) file.

    `$ conda env create -f environment.yml`

2. Activate the environment after the installation is completed.

    `$ conda activate bayesian-hierarchical-clustering`

3. Install the following libraries: `pytest`, `coverage` and `flake8`.

    `$ conda install pytest coverage flake8`

4. Run the `build_dev.sh` script.

    `$ ./build_dev.sh`

The files will be available in the `dist` folder.

## Examples

Examples can be found at [caponetto/bayesian-hierarchical-clustering-examples](https://github.com/caponetto/bayesian-hierarchical-clustering-examples).


## Contribute
All contributions are welcome, so don't hesitate to submit a pull request. ;-)

## License
This code is released under GPL 3.0 License.

Check [LICENSE](LICENSE) file for more information.
