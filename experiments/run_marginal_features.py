from sklearn.model_selection import train_test_split
from lassonet import LassoNetClassifier
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

lasso_model_marginal = LassoNetClassifier(M=10, hidden_dims=hidden_dim, verbose=1, torch_seed=seed, random_state=seed, device=device, n_iters=n_epochs, batch_size=batch_size)
lasso_model_marginal_sparse = LassoNetClassifier(M=10, hidden_dims=hidden_dim, verbose=1, torch_seed=seed, random_state=seed, device=device, n_iters=n_epochs)
marginal_feature = MarginalFeatures(X_train, y_train, lassonet=lasso_model_marginal)
marginal_feature.run_helper(lasso_model_marginal, lasso_model_marginal_sparse, X_val, y_val, X_test, y_test)
test = 1