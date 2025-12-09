## main.py
import sys
import yaml
import time
import numpy as np
import os
from dataset_loader import DatasetLoader
from model import SurrogateObjectiveGP
from constraint_classifier import NeuralEnsembleConstraintClassifier
from boundary_bounds import BoundaryBounds
from acquisition_optimizer import AcquisitionOptimizer
from evaluation import Evaluation

def main():
    # Load configuration from 'config.yaml'
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Extract configs and set defaults
    training_config = config.get('training', {})
    model_config = config.get('model', {})
    optimizer_config = config.get('optimizer', {})
    acq_config = config.get('acquisition', {})
    boundary_config = config.get('boundary', {})
    eval_config = config.get('evaluation', {})

    total_evals = eval_config.get('total_evaluations', 200)
    init_samples = eval_config.get('initial_samples', 10)
    seeds = eval_config.get('seeds', [42, 123, 456, 789, 101112, 131415, 161718, 192021, 222324, 252627])
    num_seeds = len(seeds)

    # For reproducibility, for each seed, run full experiment
    for seed_idx, seed in enumerate(seeds):
        print(f"Running seed {seed} ({seed_idx+1}/{num_seeds})")
        np.random.seed(seed)

        # Initialize dataset loader: define problem bounds (example for synthetic; user should adapt)
        # For demonstration, we assume 'dataset_loader.py' handles problem bounds internally.
        # Replace the bounds dict with the appropriate bounds for your problem.
        # For synthetic, set bounds accordingly, e.g.,
        # bounds = {'x1': (-2.25, 2.25), 'x2': (-2.5, 1.75)} for Townsend
        # For real-world, load actual bounds.
        # Here, dummy bounds are used; adapt as necessary.
        bounds = {
            'x1': (-2.25, 2.25, 'float'),
            'x2': (-2.5, 1.75, 'float')
        }
        dim = len(bounds)
        dataset = DatasetLoader(bounds, init_samples, seed=seed)

        # Initialize surrogate model for objective
        gp_hyperparams = {'lengthscale': 1.0}
        surrogate_gp = SurrogateObjectiveGP(kernel_type='RBF', hyperparameters=gp_hyperparams)

        # Prepare datasets for model training
        X_init, Y_init, labels_init = dataset.get_dataset()
        # Filter only feasible points for objective GP
        feasible_mask = np.array(labels_init) == 1
        X_feasible = X_init[feasible_mask]
        Y_feasible = np.array(Y_init)[feasible_mask]
        # Train GP surrogate
        if len(X_feasible) > 0:
            surrogate_gp.fit(X_feasible, Y_feasible)
        else:
            print("Warning: No feasible initial samples to train GP.")

        # Initialize neural ensemble classifier for constraints
        ensemble_size = model_config.get('ensemble_size', 5)
        hidden_layers = model_config.get('hidden_layers', 2)
        neurons_per_layer = model_config.get('neurons_per_layer', 64)
        classifier = NeuralEnsembleConstraintClassifier(
            input_dim=dim,
            ensemble_size=ensemble_size,
            hidden_layers=hidden_layers,
            neurons_per_layer=neurons_per_layer,
            learning_rate=training_config.get('learning_rate', 1e-3),
            epochs=training_config.get('epochs', 50),
            batch_size=training_config.get('batch_size', 32),
            device=torch.device('cpu')
        )
        # Train classifier on initial data
        X_all, Y_obj, feas_labels = dataset.get_dataset()
        classifier.train(X_all, feas_labels)

        # Initialize boundary bounds component
        boundary_bounds = BoundaryBounds(margin=boundary_config.get('margin', 0.5))

        # Prepare acquisition optimizer
        acq_func_name = acq_config.get('function', 'EI')
        alpha = acq_config.get('alpha', 1.0)
        # Based on the chosen acquisition function, define a call
        # For simplicity, only EI or UCB
        def acq_function(x: np.ndarray):
            # x shape: (d,)
            # Use surrogate for objective and last trained GP
            mean, var = surrogate_gp.predict(x.reshape(1, -1))
            std = np.sqrt(var)
            f_best = np.nanmin(Y_obj) if len(Y_obj) > 0 else np.inf
            # Expected Improvement calculation
            from scipy.stats import norm
            Z = (np.min(Y_obj) - mean) / (std + 1e-9)
            ei = (np.min(Y_obj) - mean) * norm.cdf(Z) + std * norm.pdf(Z)
            return ei.squeeze()
        def ucb_acq(x):
            mean, var = surrogate_gp.predict(x.reshape(1, -1))
            std = np.sqrt(var)
            return mean.squeeze() + alpha * std

        if acq_func_name.upper() == 'UCB':
            acq_func = ucb_acq
        else:
            acq_func = acq_function

        # Prepare optimizer
        acq_optimizer = AcquisitionOptimizer(
            acq_func=acq_func,
            C_func=None,  # We'll handle constraint coupling inside the optimizer
            bounds=bounds,
            classifier=classifier,
            boundary_bounds=boundary_bounds,
            method='differential_evolution',
            seed=seed
        )

        # Initialize evaluation metrics
        eval_times = []
        start_time = time.time()

        # For visualization and metrics
        evaluation = Evaluation(config, dataset, classifier, surrogate_gp, seeds)

        # Main BO loop
        for iteration in range(total_evals):
            # Step 1: Compute boundary bounds
            # Since optimization runs internally, we need to define the constrained function that includes boundary bounds
            def constraint_cp(x):
                # Get classifier probability and uncertainty
                C_x = classifier.predict_proba(x.reshape(1, -1))[0]
                sigma_e = classifier.predict_uncertainty(x.reshape(1, -1))[0]
                l_x = boundary_bounds.compute_lower_bound(sigma_e)
                return C_x - l_x  # constraint >=0, with boundary at C_x >= l_x
            # Prepare constraint dict for scipy optimizer
            cons = [{'type': 'ineq', 'fun': constraint_cp}]
            # Optimize acquisition with constraints
            x_next, acq_value = acq_optimizer.optimize()
            # Evaluate at x_next
            eval_start = time.time()
            f_eval, feasible = dataset.get_evaluations(x_next)
            eval_end = time.time()
            eval_time = eval_end - eval_start
            eval_times.append(eval_time)

            # Append to dataset
            dataset.X = np.vstack([dataset.X, x_next])
            if feasible:
                Y_obj_new, _ = dataset.get_evaluations(x_next)
                # True objective already evaluated inside get_evaluations()
                new_f = f_eval
            else:
                # No objective to evaluate if infeasible
                new_f = np.nan

            # Append data to arrays
            Y_obj = np.array(Y_obj)
            if feasible:
                Y_obj = np.append(Y_obj, new_f)
            else:
                Y_obj = np.append(Y_obj, np.nan)
            labels_init = np.array(feas_labels)
            labels_init = np.append(labels_init, 1 if feasible else 0)

            # Step 2: Retrain surrogate GP on feasible points
            feasible_mask = np.array(labels_init) == 1
            X_feasible = dataset.X[feasible_mask]
            if len(X_feasible) > 0:
                # Refit GP
                surrogate_gp.fit(X_feasible, Y_obj[feasible_mask])
            else:
                print("Warning: no feasible points for GP update.")

            # Step 3: Retrain classifier on all data
            classifier.train(dataset.X, feas_labels)

            # Step 4: Log and evaluate metrics
            # Get current best objective from feasible data
            if len(Y_obj[feasible_mask]) > 0:
                current_best = np.nanmin(Y_obj[feasible_mask])
            else:
                current_best = np.inf
            evaluation.compute_best_feasible(Y_obj, feas_labels, seed_idx)
            # Classifier accuracy
            pred_probs = classifier.predict_proba(dataset.X)
            true_labels = feas_labels
            acc = evaluation.compute_classifier_accuracy(true_labels, pred_probs)
            evaluation.record_evaluation(seed_idx, iteration+1, Y_obj, feas_labels, eval_times)

            # Visualization (optional, can be toggled)
            if (iteration+1) % 50 == 0 or iteration == total_evals - 1:
                # For 2D problems, plot boundary evolution
                if len(dataset.X[0]) == 2:
                    boundary_points = dataset.X[feas_labels==1]
                    evaluation.plot_boundary_evolution(seed_idx, [boundary_points])

        # Finalize after all iterations
        # Save metrics and plots
        evaluation.save_metrics_summary()
        evaluation.generate_all_plots()

        total_time_elapsed = time.time() - start_time
        print(f"Seed {seed} completed in {total_time_elapsed:.2f} seconds.")

if __name__ == '__main__':
    main()
