import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from models.model import SemiSupervisedModel, CombinedDataLoader, train_generative_model, train_supervised_logistic_regression, train_all_labels_known
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from utils.helper_functions import generate_data_with_outliers, generate_uninformative_data, plot_results_misspecified, generate_correct_data, plot_results, cross_validate_hyperparameters


def test_misspecified_joint_density_model(batch_size = 10, epochs = 1):
    """
    Test function for the misspecified joint density model experiments.
    This function generates data with outliers and uninformative unlabeled data,
    trains the models, and plots the results for Figure 2.
    """
    ratios = [1, 3, 10, 30, 100]  # Ratios of unlabeled to labeled data
    a = 0.23  # Fixed parameter for data generation
    batch_size = batch_size
    epochs = epochs
    # Experiment with Outliers
    errors_outliers = {"ME": [], "GMM": [], "Supervised": [], "All Known": []}
    for ratio in ratios:
        # Generate data with outliers
        number_labeled = 50 
        n_unlabeled = number_labeled * ratio
        n_total = number_labeled + n_unlabeled
        step_per_epoch = n_total // batch_size
        total_steps = epochs * step_per_epoch
        x_labeled, y_labeled, z_labeled, x_unlabeled, z_unlabeled, y = generate_data_with_outliers(number_labeled, n_unlabeled, a=a)
        x_val, y_val, _, _, _, _ = generate_data_with_outliers(10000, 0, a=a)

        # Train models
        ssl_model = SemiSupervisedModel(input_dim=50, output_dim=2)
        labeled_loader = DataLoader(TensorDataset(x_labeled, y_labeled, z_labeled), batch_size=batch_size, shuffle=True)
        unlabeled_loader = DataLoader(TensorDataset(x_unlabeled, z_unlabeled), batch_size=batch_size, shuffle=True)
        combined_loader = CombinedDataLoader(labeled_loader, unlabeled_loader)
        val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False)

        trainer = pl.Trainer(max_epochs=epochs, accelerator="gpu" if torch.cuda.is_available() else "cpu")
        trainer.fit(ssl_model, combined_loader, val_loader)

        me_accuracy = trainer.logged_metrics["val_acc_epoch"]
        gmm_accuracy = train_generative_model(x_labeled.numpy(), y_labeled.numpy(), x_unlabeled.numpy(), x_val.numpy(), y_val.numpy(), a, max_iter=100, case=2)
        supervised_accuracy = train_supervised_logistic_regression(x_labeled.numpy(), y_labeled.numpy(), x_val.numpy(), y_val.numpy(), max_iter= 100)
        
        # Calculate n_labeled and pass it to train_all_labels_known
        n_labeled = len(y_labeled)
        all_known_accuracy = train_all_labels_known(
            x_labeled.numpy(), y_labeled.numpy(), x_unlabeled.numpy(), y[n_labeled:n_labeled + 50 * ratio], x_val.numpy(), y_val.numpy(), max_iter= 100
        )
        
        # Store errors
        errors_outliers["ME"].append(1 - me_accuracy)
        errors_outliers["GMM"].append(1 - gmm_accuracy)
        errors_outliers["Supervised"].append(1 - supervised_accuracy)
        errors_outliers["All Known"].append(1 - all_known_accuracy)

    # Experiment with Uninformative Unlabeled Data
    errors_uninformative = {"ME": [], "GMM": [], "Supervised": [], "All Known": []}
    for ratio in ratios:
        # Generate uninformative unlabeled data
        number_labeled = 50 
        n_unlabeled = number_labeled * ratio
        n_total = number_labeled + n_unlabeled
        step_per_epoch = n_total // batch_size
        total_steps = epochs * step_per_epoch
        x_labeled, y_labeled, z_labeled, x_unlabeled, z_unlabeled, y = generate_uninformative_data(number_labeled, n_unlabeled, a=a)
        x_val, y_val, _, _, _, _ = generate_uninformative_data(10000, 0, a=a)

        # Train models
        ssl_model = SemiSupervisedModel(input_dim=50, output_dim=2)
        labeled_loader = DataLoader(TensorDataset(x_labeled, y_labeled, z_labeled), batch_size=batch_size, shuffle=True)
        unlabeled_loader = DataLoader(TensorDataset(x_unlabeled, z_unlabeled), batch_size=batch_size, shuffle=True)
        combined_loader = CombinedDataLoader(labeled_loader, unlabeled_loader)
        val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False)

        trainer = pl.Trainer(max_epochs=epochs, accelerator="gpu" if torch.cuda.is_available() else "cpu")
        trainer.fit(ssl_model, combined_loader, val_loader)

        me_accuracy = trainer.logged_metrics["val_acc_epoch"]
        gmm_accuracy = train_generative_model(x_labeled.numpy(), y_labeled.numpy(), x_unlabeled.numpy(), x_val.numpy(), y_val.numpy(), a, max_iter=100, case=2)
        supervised_accuracy = train_supervised_logistic_regression(x_labeled.numpy(), y_labeled.numpy(), x_val.numpy(), y_val.numpy(),  max_iter=100)
        
        # Calculate n_labeled and pass it to train_all_labels_known
        n_labeled = len(y_labeled)
        all_known_accuracy = train_all_labels_known(
            x_labeled.numpy(), y_labeled.numpy(), x_unlabeled.numpy(), y[n_labeled:n_labeled + 50 * ratio], x_val.numpy(), y_val.numpy(), max_iter=100
        )
        
        # Store errors
        errors_uninformative["ME"].append(1 - me_accuracy)
        errors_uninformative["GMM"].append(1 - gmm_accuracy)
        errors_uninformative["Supervised"].append(1 - supervised_accuracy)
        errors_uninformative["All Known"].append(1 - all_known_accuracy)

    # Plot results
    plot_results_misspecified(errors_outliers, errors_uninformative, ratios)



def test_correct_joint_density_model():
    bayes_errors = [1, 2.5, 5, 10, 20]  # Bayes error rates as percentages
    ratios = [1, 3, 10, 30, 100]  # Ratios of unlabeled to labeled data

    errors_bayes = {"ME": [], "GMM": [], "Supervised": [], "All Known": []}
    errors_ratios = {"ME": [], "GMM": [], "Supervised": [], "All Known": []}

    epochs = 5
    batch_size = 32
    lambda_values = [0.1, 0.2,  0.5, 1.0]  # Candidate lambda values
    weight_decay_values = [1e-5, 1e-4, 1e-3]  # Candidate weight decay values

    # Test Error vs. Bayes Error
    for bayes_error in bayes_errors:
        a = 0.23 * (7.7 / bayes_error)  # Correct scaling of a
        number_labeled = 50 
        n_unlabeled = 500
        n_total = number_labeled + n_unlabeled
        step_per_epoch = n_total // batch_size
        total_steps = epochs * step_per_epoch
        x_labeled, y_labeled, z_labeled, x_unlabeled, z_unlabeled, y = generate_correct_data(number_labeled, n_unlabeled, a=a)
        x_val, y_val, _, _, _, _ = generate_correct_data(10000, 0, a=a)

        # Train models
        
        labeled_loader = DataLoader(TensorDataset(x_labeled, y_labeled, z_labeled), batch_size=batch_size, shuffle=True)
        unlabeled_loader = DataLoader(TensorDataset(x_unlabeled, z_unlabeled), batch_size=batch_size, shuffle=True)
        combined_loader = CombinedDataLoader(labeled_loader, unlabeled_loader)
        val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False)
        ssl_model = SemiSupervisedModel(input_dim=50, output_dim=2)
        best_lambda, best_weight_decay = cross_validate_hyperparameters(
            TensorDataset(x_labeled, y_labeled, z_labeled), TensorDataset(x_unlabeled, z_unlabeled), input_dim=50, output_dim=2, 
            lambda_values=lambda_values, weight_decay_values=weight_decay_values
        )
        ssl_model.lambda_entropy = best_lambda
        ssl_model.weight_decay = best_weight_decay

        trainer = pl.Trainer(max_epochs=epochs, accelerator="gpu" if torch.cuda.is_available() else "cpu")
        trainer.fit(ssl_model, combined_loader, val_loader)

        me_accuracy = trainer.logged_metrics["val_acc_epoch"]
        gmm_accuracy = train_generative_model(x_labeled.numpy(), y_labeled.numpy(), x_unlabeled.numpy(), x_val.numpy(), y_val.numpy(), a, max_iter=total_steps)
        supervised_accuracy = train_supervised_logistic_regression(x_labeled.numpy(), y_labeled.numpy(), x_val.numpy(), y_val.numpy(), max_iter=total_steps)
        
        # Calculate n_labeled and pass it to train_all_labels_known
        n_labeled = len(y_labeled)
        all_known_accuracy = train_all_labels_known(
            x_labeled.numpy(), y_labeled.numpy(), x_unlabeled.numpy(), y[n_labeled:n_labeled + 500], x_val.numpy(), y_val.numpy(), max_iter=total_steps
        )
        
        errors_bayes["ME"].append(1 - me_accuracy)
        errors_bayes["GMM"].append(1 - gmm_accuracy)
        errors_bayes["Supervised"].append(1 - supervised_accuracy)
        errors_bayes["All Known"].append(1 - all_known_accuracy)
        

    # Test Error vs. nu/nl Ratios
    for ratio in ratios:
        a = 0.23 * (7.7 / bayes_error)  # Correct scaling of a
        number_labeled = 50 
        n_unlabeled = number_labeled * ratio
        n_total = number_labeled + n_unlabeled
        step_per_epoch = n_total // batch_size
        total_steps = epochs * step_per_epoch
        x_labeled, y_labeled, z_labeled, x_unlabeled, z_unlabeled, y = generate_correct_data(number_labeled, n_unlabeled)
        x_val, y_val, _, _, _, _ = generate_correct_data(10000, 0)

        ssl_model = SemiSupervisedModel(input_dim=50, output_dim=2)
        labeled_loader = DataLoader(TensorDataset(x_labeled, y_labeled, z_labeled), batch_size=batch_size, shuffle=True)
        unlabeled_loader = DataLoader(TensorDataset(x_unlabeled, z_unlabeled), batch_size=batch_size, shuffle=True)
        combined_loader = CombinedDataLoader(labeled_loader, unlabeled_loader)
        val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False)
        best_lambda, best_weight_decay = cross_validate_hyperparameters(
            TensorDataset(x_labeled, y_labeled, z_labeled), TensorDataset(x_unlabeled, z_unlabeled), input_dim=50, output_dim=2, 
            lambda_values=lambda_values, weight_decay_values=weight_decay_values
        )
        ssl_model.lambda_entropy = best_lambda
        ssl_model.weight_decay = best_weight_decay
        trainer = pl.Trainer(max_epochs=epochs, accelerator="gpu" if torch.cuda.is_available() else "cpu")
        trainer.fit(ssl_model, combined_loader, val_loader)

        me_accuracy = trainer.logged_metrics["val_acc_epoch"]
        gmm_accuracy = train_generative_model(x_labeled.numpy(), y_labeled.numpy(), x_unlabeled.numpy(), x_val.numpy(), y_val.numpy(), a,  max_iter=total_steps)
        supervised_accuracy = train_supervised_logistic_regression(x_labeled.numpy(), y_labeled.numpy(), x_val.numpy(), y_val.numpy(),  max_iter=total_steps)
        
        # Calculate n_labeled and pass it to train_all_labels_known
        n_labeled = len(y_labeled)
        all_known_accuracy = train_all_labels_known(
            x_labeled.numpy(), y_labeled.numpy(), x_unlabeled.numpy(), y[n_labeled:n_labeled + 50 * ratio], x_val.numpy(), y_val.numpy(),  max_iter=total_steps
        )
        
        errors_ratios["ME"].append(1 - me_accuracy)
        errors_ratios["GMM"].append(1 - gmm_accuracy)
        errors_ratios["Supervised"].append(1 - supervised_accuracy)
        errors_ratios["All Known"].append(1 - all_known_accuracy)

    plot_results(errors_bayes, errors_ratios, bayes_errors, ratios)
