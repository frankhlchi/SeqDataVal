from opendataval.dataval.api import DataEvaluator, ModelMixin
import numpy as np
import torch
from torch.utils.data import Subset
from scipy.spatial.distance import cdist
import tqdm
from sklearn.base import clone


class BipartiteMatchingEvaluator(DataEvaluator, ModelMixin):
    """Class-aware greedy bipartite matching data evaluator.
    
    This implementation combines:
    1. automatic quantile-threshold calibration,
    2. class-aware greedy matching, and
    3. feature similarity with label consistency.
    """
    
    def __init__(self, n_samples=10, threshold_range=None, random_state=None):
        super().__init__(random_state=random_state)
        
        self.n_samples = n_samples
        if threshold_range is None:
            self.threshold_range = np.linspace(0.5, 0.95, 1000)
        else:
            self.threshold_range = threshold_range
            
        self.optimal_threshold = None

    def _get_class_labels(self, y):
        """Return one-dimensional class labels."""
        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()
        if len(y.shape) > 1 and y.shape[1] > 1:
            return np.argmax(y, axis=1)
        return y.ravel()

    def _compute_similarity_matrix(self, x_train, x_valid):
        """Compute the train-validation similarity matrix."""
        if isinstance(x_train, torch.Tensor):
            x_train = x_train.cpu().numpy()
        if isinstance(x_valid, torch.Tensor):    
            x_valid = x_valid.cpu().numpy()
            
        distances = cdist(x_train, x_valid, 'euclidean')
        distances = distances / distances.std()
        similarities = np.exp(-distances**2/2)
        
        return similarities
    
    def _precompute_similarity_distribution(self, x_train, x_valid, train_labels, valid_labels):
        """Precompute the intra-class similarity distribution."""
        similarities = self._compute_similarity_matrix(x_train, x_valid)
        
        class_mask = train_labels.reshape(-1, 1) == valid_labels.reshape(1, -1)
        intra_class_similarities = similarities[class_mask]
        
        self.base_similarity_matrix = similarities
        self.intra_class_similarities = intra_class_similarities
        
        return similarities

    def _compute_valid_edges(self, quantile, train_labels, valid_labels):
        """Compute valid bipartite edges from a similarity quantile.
        
        Args:
            quantile: Similarity quantile threshold.
            train_labels: Training labels.
            valid_labels: Validation labels.
            
        Returns:
            Boolean matrix indicating valid train-validation edges.
        """
        if not hasattr(self, 'base_similarity_matrix'):
            raise ValueError("_precompute_similarity_distribution must be called first")
            
        threshold = np.quantile(self.intra_class_similarities, quantile)
        
        n_train = len(train_labels)
        n_valid = len(valid_labels)
        valid_edges = np.zeros((n_train, n_valid), dtype=bool)
        
        for i in range(n_train):
            same_class = train_labels[i] == valid_labels
            high_similarity = self.base_similarity_matrix[i] >= threshold
            valid_edges[i] = same_class & high_similarity
            
        return valid_edges

    def _compute_greedy_coverage(self, valid_edges):
        """Compute the greedy maximum-coverage ordering."""
        n_train, n_valid = valid_edges.shape
        coverage_sequence = []
        remaining_edges = valid_edges.copy()
        remaining_valid = np.ones(n_valid, dtype=bool)
        
        while True:
            coverage_counts = (remaining_edges & remaining_valid).sum(axis=1)
            
            if coverage_counts.max() == 0:
                remaining_points = set(range(n_train)) - set(coverage_sequence)
                remaining_list = list(remaining_points)
                self.random_state.shuffle(remaining_list)
                coverage_sequence.extend(remaining_list)
                break
                
            best_point = np.argmax(coverage_counts)
            coverage_sequence.append(best_point)
            
            newly_covered = remaining_edges[best_point] & remaining_valid
            remaining_valid[newly_covered] = False
            remaining_edges[:, newly_covered] = False
            
        return coverage_sequence

    def _generate_subsets_and_accuracies(self, x_train, y_train, x_valid, y_valid):
        """Generate calibration subsets and their validation accuracies."""
        print("Generating calibration subsets and validation accuracies...")
        
        ratios = self.random_state.uniform(0.01, 0.99, self.n_samples)
        
        subsets = []
        accuracies = []
        
        for ratio in tqdm.tqdm(ratios):
            size = max(1, int(ratio * len(x_train)))
            indices = self.random_state.choice(len(x_train), size, replace=False)
            
            subset_model = self.pred_model.clone()
            subset_model.fit(
                Subset(x_train, indices),
                Subset(y_train, indices)
            )
            y_pred = subset_model.predict(x_valid)
            val_acc = self.evaluate(y_valid, y_pred)
            
            subsets.append(indices)
            accuracies.append(val_acc)
            
        return subsets, accuracies

    def _find_optimal_threshold(self, x_train, y_train, x_valid, y_valid):
        """Find the optimal threshold by grid search in quantile space."""
        train_labels = self._get_class_labels(y_train)
        valid_labels = self._get_class_labels(y_valid)
        
        self._precompute_similarity_distribution(x_train, x_valid, train_labels, valid_labels)
        
        subsets, accuracies = self._generate_subsets_and_accuracies(
            x_train, y_train, x_valid, y_valid
        )
        
        print("Searching for the optimal quantile threshold...")
        best_threshold = None
        min_error = float('inf')
        
        for quantile in tqdm.tqdm(self.threshold_range):
            mse = 0
            valid_edges = self._compute_valid_edges(
                quantile, train_labels, valid_labels
            )
            
            for subset_idx, (indices, true_acc) in enumerate(zip(subsets, accuracies)):
                valid_edges_subset = valid_edges[indices]
                coverage = valid_edges_subset.any(axis=0).mean()
                mse += (coverage - true_acc) ** 2
                
            avg_mse = mse / len(subsets)
            if avg_mse < min_error:
                min_error = avg_mse
                best_threshold = quantile
                
        return best_threshold, min_error

    def train_data_values(self, *args, **kwargs):
        """Train the data-value evaluator."""
        print(
            f"Calibrating quantile threshold "
            f"({self.n_samples} random subsets of different sizes)..."
        )
        self.optimal_threshold, self.validation_error = self._find_optimal_threshold(
            self.x_train, self.y_train,
            self.x_valid, self.y_valid
        )
        print(
            f"Optimal quantile threshold: {self.optimal_threshold:.3f}, "
            f"validation MSE: {self.validation_error:.3f}"
        )
        
        train_labels = self._get_class_labels(self.y_train)
        valid_labels = self._get_class_labels(self.y_valid)
        
        valid_edges = self._compute_valid_edges(
            self.optimal_threshold,
            train_labels,
            valid_labels
        )
        
        self.coverage_sequence = self._compute_greedy_coverage(valid_edges)
        
        n_train = len(self.x_train)
        self.data_values = np.zeros(n_train)
        for i, idx in enumerate(self.coverage_sequence):
            self.data_values[idx] = n_train - i
        return self
        
    def evaluate_data_values(self) -> np.ndarray:
        """Return normalized data values."""
        normalized_values = (self.data_values - self.data_values.min()) 
        if self.data_values.max() > self.data_values.min():
            normalized_values /= (self.data_values.max() - self.data_values.min())
        return normalized_values
