import torch
import numpy as np

def entropy_loss(outputs):
    log_probs = torch.log(outputs + 1e-10)  # Avoid log(0)
    entropy = -torch.sum(outputs * log_probs, dim=1)
    return torch.mean(entropy)


def generate_data_with_outliers(n_labeled, n_unlabeled, input_dim=50, a=0.23, seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)

    mean1 = np.full(input_dim, a)
    mean2 = np.full(input_dim, -a)
    cov = np.eye(input_dim)
    cov_outliers = 100 * np.eye(input_dim)  # Large variance for outliers

    n1_labeled = n_labeled // 2
    n1_unlabeled = n_unlabeled // 2
    n1 = n1_labeled + n1_unlabeled

    # Generate main component (98% of data)
    x1_main = np.random.multivariate_normal(mean1, cov, int(0.98 * n1))
    x2_main = np.random.multivariate_normal(mean2, cov, int(0.98 * n1))

    # Generate outliers (2% of data)
    x1_outliers = np.random.multivariate_normal(mean1, cov_outliers, n1 - int(0.98 * n1))
    x2_outliers = np.random.multivariate_normal(mean2, cov_outliers, n1 - int(0.98 * n1))

    # Combine main component and outliers
    x1 = np.vstack((x1_main, x1_outliers))
    x2 = np.vstack((x2_main, x2_outliers))

    y1 = np.zeros(len(x1), dtype=int)
    y2 = np.ones(len(x2), dtype=int)

    x = np.vstack((x1, x2))
    y = np.hstack((y1, y2))

    indices = np.arange(len(x))
    np.random.shuffle(indices)
    x, y = x[indices], y[indices]

    # Create dummy variable z for labeled and unlabeled data
    z_labeled = np.zeros((n_labeled, 2))  # 2 classes
    z_labeled[np.arange(n_labeled), y[:n_labeled]] = 1  # One-hot encoding for labeled data

    z_unlabeled = np.ones((n_unlabeled, 2))  # All entries are 1 for unlabeled data

    x_labeled = torch.tensor(x[:n_labeled], dtype=torch.float32)
    y_labeled = torch.tensor(y[:n_labeled], dtype=torch.long)
    z_labeled = torch.tensor(z_labeled, dtype=torch.float32)
    x_unlabeled = torch.tensor(x[n_labeled:n_labeled + n_unlabeled], dtype=torch.float32)
    z_unlabeled = torch.tensor(z_unlabeled, dtype=torch.float32)

    return x_labeled, y_labeled, z_labeled, x_unlabeled, z_unlabeled, y


def generate_uninformative_data(n_labeled, n_unlabeled, input_dim=50, a=0.23, seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)

    mean1 = np.full(input_dim, a)
    mean2 = np.full(input_dim, -a)
    cov = np.eye(input_dim)

    n1_labeled = n_labeled // 2
    n1_unlabeled = n_unlabeled // 2
    n1 = n1_labeled + n1_unlabeled

    x1 = np.random.multivariate_normal(mean1, cov, n1)
    x2 = np.random.multivariate_normal(mean2, cov, n1)

    # Assign labels based on x2 > x1
    y1 = (x1[:, 1] > x1[:, 0]).astype(int)
    y2 = (x2[:, 1] > x2[:, 0]).astype(int)

    x = np.vstack((x1, x2))
    y = np.hstack((y1, y2))

    indices = np.arange(len(x))
    np.random.shuffle(indices)
    x, y = x[indices], y[indices]

    # Create dummy variable z for labeled and unlabeled data
    z_labeled = np.zeros((n_labeled, 2))  # 2 classes
    z_labeled[np.arange(n_labeled), y[:n_labeled]] = 1  # One-hot encoding for labeled data

    z_unlabeled = np.ones((n_unlabeled, 2))  # All entries are 1 for unlabeled data

    x_labeled = torch.tensor(x[:n_labeled], dtype=torch.float32)
    y_labeled = torch.tensor(y[:n_labeled], dtype=torch.long)
    z_labeled = torch.tensor(z_labeled, dtype=torch.float32)
    x_unlabeled = torch.tensor(x[n_labeled:n_labeled + n_unlabeled], dtype=torch.float32)
    z_unlabeled = torch.tensor(z_unlabeled, dtype=torch.float32)

    return x_labeled, y_labeled, z_labeled, x_unlabeled, z_unlabeled, y


import matplotlib.pyplot as plt
def plot_results_misspecified(errors_outliers, errors_uninformative, ratios):
    plt.figure(figsize=(12, 5))

    # Plot Test Error vs. nu/nl Ratio (Outliers)
    plt.subplot(1, 2, 1)
    plt.plot(ratios, errors_outliers["ME"], 'o-', label="Minimum Entropy (◦)")
    plt.plot(ratios, errors_outliers["GMM"], '+-', label="GMM (+)")
    plt.plot(ratios, errors_outliers["Supervised"], 'k--', label="Supervised (dashed)")
    plt.plot(ratios, errors_outliers["All Known"], 'k-.', label="All Labels Known (dash-dotted)")
    plt.xlabel("Ratio $n_u / n_l$")
    plt.ylabel("Test Error (%)")
    plt.title("Outliers Experiment")
    plt.legend()

    # Plot Test Error vs. nu/nl Ratio (Uninformative Unlabeled Data)
    plt.subplot(1, 2, 2)
    plt.plot(ratios, errors_uninformative["ME"], 'o-', label="Minimum Entropy (◦)")
    plt.plot(ratios, errors_uninformative["GMM"], '+-', label="GMM (+)")
    plt.plot(ratios, errors_uninformative["Supervised"], 'k--', label="Supervised (dashed)")
    plt.plot(ratios, errors_uninformative["All Known"], 'k-.', label="All Labels Known (dash-dotted)")
    plt.xlabel("Ratio $n_u / n_l$")
    plt.ylabel("Test Error (%)")
    plt.title("Uninformative Unlabeled Data Experiment")
    plt.legend()

    plt.tight_layout()
    plt.show()



def plot_results(errors_bayes, errors_ratios, bayes_errors, ratios):
    plt.figure(figsize=(10, 5))

    # Plot Test Error vs. Bayes Error
    plt.subplot(1, 2, 1)
    plt.plot(bayes_errors, errors_bayes["ME"], 'o-', label="Minimum Entropy (◦)")
    plt.plot(bayes_errors, errors_bayes["GMM"], '+-', label="GMM (+)")
    plt.plot(bayes_errors, errors_bayes["Supervised"], 'k--', label="Supervised (dashed)")
    plt.plot(bayes_errors, errors_bayes["All Known"], 'k-.', label="All Labels Known (dash-dotted)")
    plt.xlabel("Bayes Error (%)")
    plt.ylabel("Test Error (%)")
    plt.legend()

    # Plot Test Error vs. nu/nl Ratios
    plt.subplot(1, 2, 2)
    plt.plot(ratios, errors_ratios["ME"], 'o-', label="Minimum Entropy (◦)")
    plt.plot(ratios, errors_ratios["GMM"], '+-', label="GMM (+)")
    plt.plot(ratios, errors_ratios["Supervised"], 'k--', label="Supervised (dashed)")
    plt.plot(ratios, errors_ratios["All Known"], 'k-.', label="All Labels Known (dash-dotted)")
    plt.xlabel("Ratio $n_u / n_l$")
    plt.ylabel("Test Error (%)")
    plt.legend()

    plt.tight_layout()
    plt.show()


def generate_correct_data(n_labeled, n_unlabeled, input_dim=50, a=0.23, seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)

    mean1 = np.full(input_dim, a)
    mean2 = np.full(input_dim, -a)
    cov = np.eye(input_dim)

    n1_labeled = n_labeled // 2
    n1_unlabeled = n_unlabeled // 2
    n1 = n1_labeled + n1_unlabeled

    x1 = np.random.multivariate_normal(mean1, cov, n1)
    x2 = np.random.multivariate_normal(mean2, cov, n1)

    y1 = np.zeros(len(x1), dtype=int)
    y2 = np.ones(len(x2), dtype=int)

    x = np.vstack((x1, x2))
    y = np.hstack((y1, y2))

    indices = np.arange(len(x))
    np.random.shuffle(indices)
    x, y = x[indices], y[indices]

    # Create dummy variable z for labeled and unlabeled data
    z_labeled = np.zeros((n_labeled, 2))  # 2 classes
    z_labeled[np.arange(n_labeled), y[:n_labeled]] = 1  # One-hot encoding for labeled data

    z_unlabeled = np.ones((n_unlabeled, 2))  # All entries are 1 for unlabeled data

    x_labeled = torch.tensor(x[:n_labeled], dtype=torch.float32)
    y_labeled = torch.tensor(y[:n_labeled], dtype=torch.long)
    z_labeled = torch.tensor(z_labeled, dtype=torch.float32)
    x_unlabeled = torch.tensor(x[n_labeled:n_labeled + n_unlabeled], dtype=torch.float32)
    z_unlabeled = torch.tensor(z_unlabeled, dtype=torch.float32)

    return x_labeled, y_labeled, z_labeled, x_unlabeled, z_unlabeled, y



from sklearn.model_selection import KFold
from torch.utils.data import Subset
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from models.model import SemiSupervisedModel, CombinedDataLoader
import pytorch_lightning as pl

def cross_validate_hyperparameters(labeled_dataset, unlabeled_dataset, input_dim, output_dim, lambda_values, weight_decay_values, epochs=5):
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    best_lambda = None
    best_weight_decay = None
    best_val_loss = float('inf')

    for lambda_entropy in lambda_values:
        for weight_decay in weight_decay_values:
            avg_val_loss = 0.0

            # Perform 10-fold cross-validation on labeled data
            for train_indices, val_indices in kfold.split(labeled_dataset):
                labeled_train_subset = Subset(labeled_dataset, train_indices)
                labeled_val_subset = Subset(labeled_dataset, val_indices)

                unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=32, shuffle=True)

                # Create data loaders for labeled data
                labeled_train_loader = DataLoader(labeled_train_subset, batch_size=32, shuffle=True)
                labeled_val_loader = DataLoader(labeled_val_subset, batch_size=32, shuffle=False)

                combined_loader = CombinedDataLoader(labeled_train_loader, unlabeled_loader)

                model = SemiSupervisedModel(input_dim=input_dim, output_dim=output_dim)
                model.lambda_entropy = lambda_entropy  # Set lambda_entropy
                model.weight_decay = weight_decay

                # Train the model
                trainer = pl.Trainer(max_epochs=epochs, accelerator="gpu" if torch.cuda.is_available() else "cpu")
                trainer.fit(model, combined_loader, labeled_val_loader)

                # Get the validation loss
                val_loss = trainer.logged_metrics['val_loss_epoch']
                avg_val_loss += val_loss / 10  # Average over 10 folds

            # Update best parameters if current setup is better
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_lambda = lambda_entropy
                best_weight_decay = weight_decay

    return best_lambda, best_weight_decay
