# Semi-Supervised Learning Experiments

This repository contains the implementation of semi-supervised learning experiments inspired by a research paper exploring the impact of misspecified and correctly specified joint density models. The experiments aim to compare different training approaches under various conditions, such as the presence of outliers, uninformative data, and varying ratios of labeled to unlabeled data.

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
This experiment tests the performance of a semi-supervised learning model when the data contains:
- **Outliers**: Extreme values with high variance added to the labeled and unlabeled datasets.
- **Uninformative Unlabeled Data**: Unlabeled data that does not follow the same distribution as the labeled data.

The results are plotted to visualize the effect of varying the ratio of unlabeled to labeled data on the test error rates for different models.

### 2. Correctly Specified Joint Density Model
This experiment evaluates model performance when the joint density of labeled and unlabeled data is correctly specified. The results are plotted to analyze the:
- Relationship between Bayes error rates and test error.
- Impact of varying the ratio of unlabeled to labeled data on test error rates.

## Implementation Notes

- **Partial Implementation**: Only the first two experiments from the paper are implemented.
- **Single Run per Configuration**: Unlike the paper, which reports results as the average of 10 runs on different sample sizes, this implementation performs only a single run per configuration.

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

This script invokes the `test_misspecified_joint_density_model` function from `testing.py`, which handles the data generation, training, and result visualization.

### Results

The results of the experiments are plotted:
- **Outliers and Uninformative Unlabeled Data**: Test errors for Minimum Entropy (ME), Gaussian Mixture Model (GMM), Supervised Logistic Regression, and All Labels Known.
- **Correct Joint Density Model**: Test errors against Bayes error rates and unlabeled-to-labeled data ratios.

## Future Work

To fully replicate the experiments described in the paper:
1. Implement multiple runs (e.g., 10 runs) for each configuration and compute the average results.
2. Extend the implementation to cover additional experiments from the paper.
3. Optimize model hyperparameters and training loops for larger datasets.

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

## Acknowledgments

This project is inspired by a research paper on semi-supervised learning. Special thanks to the authors for providing the theoretical framework and experiment design.
