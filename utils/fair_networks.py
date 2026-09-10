import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Function
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

# ---------------------------------------------------------------------------
# Gradient Reversal Layer
# ---------------------------------------------------------------------------
class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.lambda_
        return output, None

class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_=1.0):
        super(GradientReversalLayer, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

# ---------------------------------------------------------------------------
# 1. Fair MLP (Composite Loss with Flattened Demographic Parity)
# ---------------------------------------------------------------------------
class FairMLPModel(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64]):
        super().__init__()
        layers = []
        in_d = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_d, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            in_d = h
        layers.append(nn.Linear(in_d, 1))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x).squeeze(-1)

class FairMLPClassifier(BaseEstimator, ClassifierMixin):
    """
    MLP Classifier with Composite Loss: BCE + Demographic Parity Penalty.
    The sensitive attributes are expected to be the LAST columns of X.
    """
    def __init__(self, hidden_dims=(128, 64), lr=0.001, epochs=20, batch_size=256,
                 lambda_fairness=0.5, num_sensitive_attrs=1, sensitive_dims_list=None, random_state=42):
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.lambda_fairness = lambda_fairness
        self.num_sensitive_attrs = num_sensitive_attrs
        self.sensitive_dims_list = sensitive_dims_list
        self.random_state = random_state

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        
        # Split features and sensitive attributes
        num_features = X.shape[1] - self.num_sensitive_attrs
        X_features = np.asarray(X[:, :num_features], dtype=np.float32)
        X_sens = np.asarray(X[:, -self.num_sensitive_attrs:], dtype=int)
        
        # Flatten intersectional groups to 1D based on global sensitive_dims_list
        # If sensitive_dims_list is None, fallback to discovering it (not recommended for strict CV)
        sens_dims = self.sensitive_dims_list
        if sens_dims is None:
            sens_dims = [(X_sens[:, i].max() - X_sens[:, i].min() + 1) for i in range(self.num_sensitive_attrs)]
            
        multiplier = 1
        flattened_sens = np.zeros(X.shape[0], dtype=int)
        for i in range(self.num_sensitive_attrs):
            col = X_sens[:, i]
            col = col - col.min() # ensure 0-indexed locally
            flattened_sens += col * multiplier
            multiplier *= sens_dims[i]
            
        _, group_indices = np.unique(flattened_sens, return_inverse=True)
        # Using the product of dimensions ensures the total capacity is known,
        # but the unique actual groups are what bincount uses.
        # It's safer to use the maximum possible ID + 1 to avoid mismatch between folds.
        self.num_groups_ = np.prod(sens_dims)
        
        print(f"  [FairMLP] Assigned {len(np.unique(group_indices))} active groups out of {self.num_groups_} possible intersectional groups.")
        
        # Convert to tensors
        X_tensor = torch.tensor(X_features, dtype=torch.float32)
        y_tensor = torch.tensor(np.asarray(y, dtype=np.float32), dtype=torch.float32)
        group_tensor = torch.tensor(group_indices, dtype=torch.long)
        
        dataset = TensorDataset(X_tensor, y_tensor, group_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Initialize model
        torch.manual_seed(self.random_state)
        # Convert tuple back to list for model architecture
        hidden_dims_list = list(self.hidden_dims) if self.hidden_dims is not None else []
        self.model_ = FairMLPModel(input_dim=num_features, hidden_dims=hidden_dims_list)
        optimizer = optim.Adam(self.model_.parameters(), lr=self.lr)
        bce_loss_fn = nn.BCEWithLogitsLoss()
        
        self.model_.train()
        for epoch in range(self.epochs):
            for xb, yb, gb in loader:
                optimizer.zero_grad()
                logits = self.model_(xb)
                bce = bce_loss_fn(logits, yb)
                
                # Demographic Parity Penalty
                probs = torch.sigmoid(logits)
                global_mean = probs.mean()
                
                group_sums = torch.zeros(self.num_groups_, device=xb.device)
                group_sums.scatter_add_(0, gb, probs)
                group_counts = torch.bincount(gb, minlength=self.num_groups_).float()
                
                valid = group_counts > 0
                if valid.any():
                    group_means = group_sums[valid] / group_counts[valid]
                    dp_penalty = ((group_means - global_mean)**2).mean()
                else:
                    dp_penalty = 0.0
                    
                loss = bce + self.lambda_fairness * dp_penalty
                loss.backward()
                optimizer.step()
                
        self.is_fitted_ = True
        return self

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X, dtype=None)
        num_features = X.shape[1] - self.num_sensitive_attrs
        X_features = np.asarray(X[:, :num_features], dtype=np.float32)
        X_tensor = torch.tensor(X_features, dtype=torch.float32)
        
        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(X_tensor)
            probs = torch.sigmoid(logits).numpy()
        
        return np.vstack([1 - probs, probs]).T

    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(int)

# ---------------------------------------------------------------------------
# 2. Adversarial Fair MLP (Multitask with GRL and Multiple Heads)
# ---------------------------------------------------------------------------
class AdversarialFairModel(nn.Module):
    def __init__(self, input_dim, sensitive_dims_list, hidden_dims=[128, 64], lambda_adv=1.0):
        super().__init__()
        enc_layers = []
        in_d = input_dim
        for h in hidden_dims:
            enc_layers.append(nn.Linear(in_d, h))
            enc_layers.append(nn.ReLU())
            enc_layers.append(nn.Dropout(0.2))
            in_d = h
        self.encoder = nn.Sequential(*enc_layers)
        
        self.task_head = nn.Linear(in_d, 1)
        
        self.grl = GradientReversalLayer(lambda_=lambda_adv)
        self.adv_heads = nn.ModuleList([
            nn.Linear(in_d, out_c) for out_c in sensitive_dims_list
        ])
        
    def forward(self, x):
        z = self.encoder(x)
        logits_task = self.task_head(z).squeeze(-1)
        
        z_rev = self.grl(z)
        logits_advs = [head(z_rev) for head in self.adv_heads]
        return logits_task, logits_advs

class AdversarialFairMLPClassifier(BaseEstimator, ClassifierMixin):
    """
    Multitask Adversarial MLP. Encoder unlearns sensitive attributes using GRL.
    The sensitive attributes are expected to be the LAST columns of X.
    """
    def __init__(self, hidden_dims=(128, 64), lr=0.001, epochs=20, batch_size=256,
                 lambda_adv=1.0, num_sensitive_attrs=1, sensitive_dims_list=None, random_state=42):
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.lambda_adv = lambda_adv
        self.num_sensitive_attrs = num_sensitive_attrs
        self.sensitive_dims_list = sensitive_dims_list
        self.random_state = random_state

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        
        num_features = X.shape[1] - self.num_sensitive_attrs
        X_features = np.asarray(X[:, :num_features], dtype=np.float32)
        X_sens = np.asarray(X[:, -self.num_sensitive_attrs:], dtype=int)
        
        # Get number of classes per sensitive attribute
        self.sens_dims_ = self.sensitive_dims_list
        if self.sens_dims_ is None:
            self.sens_dims_ = []
            for i in range(self.num_sensitive_attrs):
                self.sens_dims_.append(X_sens[:, i].max() - X_sens[:, i].min() + 1)
                
        for i in range(self.num_sensitive_attrs):
            X_sens[:, i] = X_sens[:, i] - X_sens[:, i].min() # ensure 0-indexed locally
            
        X_tensor = torch.tensor(X_features, dtype=torch.float32)
        y_tensor = torch.tensor(np.asarray(y, dtype=np.float32), dtype=torch.float32)
        s_tensor = torch.tensor(X_sens, dtype=torch.long)
        
        dataset = TensorDataset(X_tensor, y_tensor, s_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        torch.manual_seed(self.random_state)
        hidden_dims_list = list(self.hidden_dims) if self.hidden_dims is not None else []
        self.model_ = AdversarialFairModel(
            input_dim=num_features, 
            sensitive_dims_list=self.sens_dims_,
            hidden_dims=hidden_dims_list,
            lambda_adv=self.lambda_adv
        )
        
        optimizer = optim.Adam(self.model_.parameters(), lr=self.lr)
        bce_loss_fn = nn.BCEWithLogitsLoss()
        ce_loss_fn = nn.CrossEntropyLoss()
        
        self.model_.train()
        for epoch in range(self.epochs):
            for xb, yb, sb in loader:
                optimizer.zero_grad()
                logits_task, logits_advs = self.model_(xb)
                
                loss_task = bce_loss_fn(logits_task, yb)
                loss_adv = 0.0
                for j in range(self.num_sensitive_attrs):
                    # sb[:, j] is the target class for adversary j
                    loss_adv += ce_loss_fn(logits_advs[j], sb[:, j])
                
                # Due to GRL, loss_adv gradients are inverted for the encoder.
                # Adding it here ensures the adversaries minimize it, while encoder maximizes it.
                loss = loss_task + loss_adv
                loss.backward()
                optimizer.step()
                
        self.is_fitted_ = True
        return self

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X, dtype=None)
        num_features = X.shape[1] - self.num_sensitive_attrs
        X_features = np.asarray(X[:, :num_features], dtype=np.float32)
        X_tensor = torch.tensor(X_features, dtype=torch.float32)
        
        self.model_.eval()
        with torch.no_grad():
            logits_task, _ = self.model_(X_tensor)
            probs = torch.sigmoid(logits_task).numpy()
        
        return np.vstack([1 - probs, probs]).T

    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(int)
