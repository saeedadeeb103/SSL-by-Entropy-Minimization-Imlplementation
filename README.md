# Semi-Supervised Learning by Entropy Minimization

This repository contains an implementation of semi-supervised learning experiments inspired by the paper "Semi-Supervised Learning by Entropy Minimization" (Grandvalet & Bengio, NeurIPS 2004). The paper introduces minimum entropy regularization as a method to leverage unlabeled data effectively in semi-supervised learning scenarios.

## Key Features

- **Minimum Entropy Regularization**: Implements entropy minimization to encourage confident predictions on unlabeled data.
- **Comparison Across Scenarios**:
  - **Misspecified Joint Density Model**: Evaluates performance when data distributions contain outliers or uninformative unlabeled data.
  - **Correctly Specified Joint Density Model**: Tests model behavior when data follow the assumed distribution.
- **Models Evaluated**:
  - Minimum Entropy (ME) logistic regression
  - Gaussian Mixture Models (GMM)
  - Supervised Logistic Regression
  - Logistic Regression with All Labels Known (upper bound)

## Folder Structure

```
├── models
│   └── model.py                # Model and utilities for semi-supervised learning
├── testing_cases
│   └── testing.py              # Test functions for the experiments
├── utils
│   └── helper_functions.py     # Helper functions for data generation and plotting
├── LICENSE                     # License information
├── README.md                   # Project description and usage guide
└── training.py                 # Main script to run the experiments
```

## Experiments Implemented

### 1. Misspecified Joint Density Model
This experiment evaluates performance in scenarios where the data contains:
- **Outliers**: High-variance noise added to the dataset.
- **Uninformative Unlabeled Data**: Unlabeled data with distributions that do not align with the labeled data.

Results include test error rates for different models as a function of the ratio of unlabeled to labeled data.

### 2. Correctly Specified Joint Density Model
This experiment examines performance when the data follows the assumed joint density distribution. The results analyze:
- **Impact of Bayes Error**: Relationship between theoretical Bayes error and observed test error.
- **Effect of Unlabeled Data**: Test error rates as the ratio of unlabeled to labeled data increases.

## Implementation Notes

- **Partial Implementation**: Only the first two experiments from the paper are implemented.
- **Single Run per Configuration**: Unlike the paper, which averages results over 10 runs with different training samples, this implementation performs a single run per configuration.

## Usage

### Prerequisites

Ensure the following libraries are installed:
- `pytorch`
- `pytorch_lightning`
- `torchmetrics`
- `numpy`
- `matplotlib`
- `scikit-learn`

You can install these dependencies using:
```bash
pip install -r requirements.txt
```

### Running the Experiments

To execute the experiments, run the `training.py` script:
```bash
python training.py
```

This script invokes the `test_misspecified_joint_density_model` function from `testing.py`, which handles data generation, training, and result visualization.

### Results

The results are plotted:
- **Misspecified Joint Density Model**: Test errors for ME, GMM, Supervised Logistic Regression, and All Labels Known as a function of unlabeled-to-labeled data ratio.
- **Correctly Specified Joint Density Model**: Test errors vs. Bayes error and unlabeled-to-labeled data ratio.

## Future Work

To fully replicate the experiments described in the paper:
1. Implement multiple runs (e.g., 10 runs) for each configuration and compute the average results.
2. Extend the implementation to cover additional experiments from the paper, such as robustness to violations of the cluster assumption.
3. Optimize model hyperparameters and expand the analysis to other datasets.

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

## References

- Grandvalet, Y., & Bengio, Y. (2004). Semi-Supervised Learning by Entropy Minimization. NeurIPS 2004. [Link to Paper](https://proceedings.neurips.cc/paper_files/paper/2004/file/96f2b50b5d3613adf9c27049b2a888c7-Paper.pdf)

## Acknowledgments

This project draws inspiration from the paper "Semi-Supervised Learning by Entropy Minimization." Special thanks to the authors for the foundational methodology and experimental framework.
