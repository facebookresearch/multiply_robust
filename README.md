# Multiply Robust Estimation


This software package implements the multiply robust method proposed in the paper "Multiply Robust Estimation for Local Distribution Shifts with Multiple Domains".

```
@inproceedings{wilkins-reeves2024multiply,
    author = {Wilkins-Reeves, Steven and Chen, Xu and Ma, Qi and Agarwal, Christine and Hofleitner, Aude},
    title = {{Multiply Robust Estimation for Local Distribution Shifts with Multiple Domains}},
    booktitle = {Proceedings of the 41st International Conference on Machine Learning. (ICML 2024)},
    year = {2024},
}
```

## Usage

* estimators: implement the multiply robust estimation method
* utils: implement utility functions including data preprocessing and segment clustering algorithms
* weights: implement different importance weighting methods to adjust for covariate shifts and label shifts

A demo notebook will be added!

## Installation

**Installation Requirements**

* Python >= 3.6
* numpy >= 1.24.4
* sklearn >= 1.2.2
* xgboost >= 1.2.0
* scipy >= 1.10.1
* cvxopt >= 1.2.7

## Contribute

See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.

## License
multiply_robust package is MIT licensed, as found in the [LICENSE](LICENSE) file.
