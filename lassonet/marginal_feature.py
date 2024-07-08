# Generic class for creating marginal features based on X and y data.
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from lassonet.utils import eval_on_path


class MarginalFeatures:
    def __init__(self, X, y, lassonet, epochs=1000):
        self.lassonet = lassonet
        self.X, self.y = X, y
        self.data_dim = X.shape[1]
        self.n_samples = X.shape[0]
        self.loss = lassonet.criterion
        self.models = []  # used for storing the models for each feature
        self.marginal_features = []
        self.epochs = epochs
        self.path = None
        self.output_shape = self.lassonet._output_shape(y)
        self.path_sparse = {}

    def create_marginal_features(self):
        """
        Calculate marginal features based on the data X and y using the loss function.
        """
        for i in range(self.data_dim):
            x_k = self.X[:, i]
            x_k = x_k.reshape(-1, 1)
            # Train the model
            if self.loss._get_name() == 'CrossEntropyLoss':
                model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
                model.fit(x_k, self.y)
                marginal_feature = model.predict_log_proba(x_k)
            else:
                model = LinearRegression()
                model.fit(x_k, self.y)
                marginal_feature = model.predict(x_k)
            self.models.append(model)
            self.marginal_features.append(marginal_feature[:, 1:])
        self.marginal_features = np.concatenate(self.marginal_features, axis=1)
        return self.marginal_features

    def create_marginal_features_nn(self):
        """
        Calculate marginal features based on the data X and y using the loss function.
        The marginal features are obtained by regress y on x_k for k = 1,..., p, and obtain
        marginal feature f_k = \\hat{E}(y|x_k)
        """
        for i in range(self.data_dim):
            x_k = self.X[:, i]
            x_k = x_k.reshape(-1, 1)
            #  define a neural network that resemble to regression then use an optimizer (like SGD or Adam) to update the model's parameters based on the gradients computed from the loss function.
            model = torch.nn.Linear(in_features=1, out_features=self.lassonet._output_shape(self.y), bias=True)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

            # Training loop
            for epoch in range(self.epochs):
                # Forward pass: compute predicted y by passing x to the model
                y_pred = model(x_k)

                # Compute the loss
                loss = self.loss(y_pred, self.y)

                # Before the backward pass, use the optimizer object to zero all of the
                # gradients for the variables it will update (which are the learnable
                # weights of the model)
                optimizer.zero_grad()

                # Backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()

                # Calling the step function on an Optimizer makes an update to its parameters
                optimizer.step()

            # Save the model
            self.models.append(model)

            marginal_feature = model(x_k).detach()  # detach() is used to remove the computational graph
            self.marginal_features.extend(marginal_feature)
        self.marginal_features = np.array(self.marginal_features).T
        return self.marginal_features

    def get_marginal_features(self, X_new):
        """
        Get the marginal features for new data X_new based on the saved models.
        """
        # check if the models are already created
        if len(self.models) == 0:
            raise ValueError("No models are created yet. Please create the marginal features first.")
        marginal_features = []
        for i in range(self.data_dim):
            x_k = X_new[:, i]
            x_k = x_k.reshape(-1, 1)
            model = self.models[i]
            if self.loss._get_name() == 'CrossEntropyLoss':
                marginal_feature = model.predict_log_proba(x_k)
            else:
                marginal_feature = model.predict(x_k)
            marginal_features.append(marginal_feature[:, 1:])
        marginal_features = np.concatenate(marginal_features, axis=1)
        return marginal_features

    def data_helper(self, X_val, X_test=None):
        """
        Helper function for creating marginal features for X_val and X_test.
        """
        X_train_margin = self.create_marginal_features()
        X_val_marginal = self.get_marginal_features(X_val)
        if X_test is not None:
            X_test_marginal = self.get_marginal_features(X_test)
        else:
            X_test_marginal = None
        return X_train_margin, X_val_marginal, X_test_marginal

    def run_helper(self, model, model_sparse, X_val, y_val, X_test=None, y_test=None, K_list=None):
        """
        Helper function for running experiments on the marginal features.
        """
        X_train_marginal, X_val_marginal, X_test_marginal = self.data_helper(X_val, X_test)
        self.path = model.path(X_train_marginal, self.y, X_val=X_val_marginal, y_val=y_val, return_state_dicts=True)
        # Select the features
        if K_list is None:
            K_list = [50, int(50 * np.log(self.output_shape)), 50 * self.output_shape, X_train_marginal.shape[1]]
        for K in K_list:
            desired_save = next(save for save in self.path if save.selected.sum().item() <= K)
            SELECTED_FEATURES = desired_save.selected
            print("Number of selected features:", SELECTED_FEATURES.sum().item())
            # Select the features from the training, validation, and test data
            X_train_selected = X_train_marginal[:, SELECTED_FEATURES]
            X_val_selected = X_val_marginal[:, SELECTED_FEATURES]
            X_test_selected = X_test_marginal[:, SELECTED_FEATURES]

            self.path_sparse[str(K)] = model_sparse.path(X_train_selected, self.y, X_val=X_val_selected, y_val=y_val,
                                                         lambda_seq=[0], return_state_dicts=True)[:1]

            # Evaluate the model on the test data
            score = eval_on_path(model_sparse, self.path_sparse[str(K)], X_test_selected, y_test)
            print("Test accuracy:", score)
