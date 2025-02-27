from sklearn.model_selection import train_test_split
from lassonet import LassoNetClassifier
from lassonet.inference_adaptive import AdaptiveLassoNetClassifier
from lassonet.marginal_feature import MarginalFeatures
from lassonet.utils import eval_on_path
from experiments.data_utils import load_dataset
import torch
import pickle

seed = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 256
K = 50 # Number of features to select
n_epochs = 1000
# dataset = 'ISOLET'
dataset = 'MICE'
# dataset = 'MNIST'
# dataset = 'Activity'

# Load dataset and split the data
(X_train_valid, y_train_valid), (X_test, y_test) = load_dataset(dataset)
X_train, X_val, y_train, y_val = train_test_split(X_train_valid, y_train_valid, test_size=0.1, random_state=seed)
print(X_train.shape)
# Set the dimensions of the hidden layers
data_dim = X_test.shape[1]
hidden_dim = (data_dim//3,)

lasso_model = LassoNetClassifier(M=10, hidden_dims=hidden_dim, verbose=2, torch_seed=seed, random_state=seed, device=device, n_iters=n_epochs, batch_size=batch_size)
path = lasso_model.path(X_train, y_train, X_val=X_val, y_val=y_val)

# Select the features
K_list = [5, 10, 15, 20]
for K in K_list:
    desired_save = next(save for save in path if save.selected.sum().item() <= K)
    SELECTED_FEATURES = desired_save.selected
    print("Number of selected features:", SELECTED_FEATURES.sum().item())

    # Select the features from the training, validation, and test data
    X_train_selected = X_train[:, SELECTED_FEATURES]
    X_val_selected = X_val[:, SELECTED_FEATURES]
    X_test_selected = X_test[:, SELECTED_FEATURES]

    # Initialize another LassoNetClassifier for retraining with the selected features
    lasso_sparse = LassoNetClassifier(M=10, hidden_dims=hidden_dim, verbose=2, torch_seed=seed, random_state=seed, device=device, n_iters=n_epochs)
    path_sparse = lasso_sparse.path(X_train_selected, y_train, X_val=X_val_selected, y_val=y_val, lambda_seq=[0], return_state_dicts=True)[:1]

    # Evaluate the model on the test data
    score = eval_on_path(lasso_sparse, path_sparse, X_test_selected, y_test)
    print(f"LassoNet K = {K}, Test accuracy:", score)

# Initialize the LassoNetClassifier model and compute the path
print("-----------------------------------")
adaptive_lasso_model = AdaptiveLassoNetClassifier(M=10, hidden_dims=hidden_dim, verbose=1, torch_seed=seed, random_state=seed, device=device, n_iters=n_epochs, batch_size=batch_size, path_multiplier=1.05)
adaptive_lasso_sparse = LassoNetClassifier(M=10, hidden_dims=hidden_dim, verbose=1, torch_seed=seed, random_state=seed, device=device, n_iters=n_epochs)
marginal_feature = MarginalFeatures(X_train, y_train, lassonet=adaptive_lasso_model)
marginal_feature.run_helper(adaptive_lasso_model, adaptive_lasso_sparse, X_val, y_val, X_test, y_test, K_list=[5, 10, 15, 20, 5 * marginal_feature.output_shape, 10 * marginal_feature.output_shape])


# lasso_model_marginal = LassoNetClassifier(M=10, hidden_dims=hidden_dim, verbose=1, torch_seed=seed, random_state=seed, device=device, n_iters=n_epochs, batch_size=batch_size)
# lasso_model_marginal_sparse = LassoNetClassifier(M=10, hidden_dims=hidden_dim, verbose=1, torch_seed=seed, random_state=seed, device=device, n_iters=n_epochs)
# marginal_feature = MarginalFeatures(X_train, y_train, lassonet=lasso_model_marginal)
# marginal_feature.run_helper(lasso_model_marginal, lasso_model_marginal_sparse, X_val, y_val, X_test, y_test)
