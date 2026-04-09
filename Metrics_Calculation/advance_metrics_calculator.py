import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from sklearn.metrics import roc_auc_score

class AdvancedMetricsCalculator:
    """YOUR EXISTING AdvancedMetricsCalculator - UNCHANGED"""
    
    @staticmethod
    def compute_uncertainty_metrics(predictions, labels, uncertainties):
        metrics = {}
        
        errors = np.abs(predictions - labels)
        total_uncertainty = uncertainties['epistemic'] + uncertainties['aleatoric']
        
        correlations = []
        for i in range(predictions.shape[1]):
            if np.std(total_uncertainty[:, i]) > 0:
                corr = np.corrcoef(total_uncertainty[:, i], errors[:, i])[0, 1]
                correlations.append(corr if not np.isnan(corr) else 0.0)
            else:
                correlations.append(0.0)
        
        metrics['uncertainty_error_correlation'] = np.mean(correlations)
        
        metrics['auupc'] = AdvancedMetricsCalculator._compute_auupc(
            predictions, labels, total_uncertainty
        )
        
        metrics['uncertainty_calibration'] = AdvancedMetricsCalculator._compute_uncertainty_calibration(
            predictions, labels, total_uncertainty
        )
        
        eps_ratio = uncertainties['epistemic'] / (total_uncertainty + 1e-8)
        metrics['epistemic_ratio_mean'] = np.mean(eps_ratio)
        metrics['epistemic_ratio_std'] = np.std(eps_ratio)
        
        return metrics

    @staticmethod
    def _compute_auupc(predictions, labels, uncertainties):
        auupc_scores = []
        
        for i in range(predictions.shape[1]):
            sorted_indices = np.argsort(uncertainties[:, i])
            
            aucs = []
            actual_fractions = []
            
            target_fractions = np.linspace(0.1, 1.0, 10)
            
            for frac in target_fractions:
                n_keep = max(2, int(frac * len(sorted_indices)))
                keep_indices = sorted_indices[:n_keep]
                
                if len(np.unique(labels[keep_indices, i])) > 1:
                    try:
                        auc = roc_auc_score(labels[keep_indices, i], predictions[keep_indices, i])
                    except:
                        auc = 0.5
                    aucs.append(auc)
                    actual_fractions.append(frac)
            
            if len(aucs) > 1:
                auupc_scores.append(np.trapz(aucs, actual_fractions))
            elif len(aucs) == 1:
                auupc_scores.append(aucs[0])
            else:
                auupc_scores.append(0.5)
        
        return np.mean(auupc_scores) if auupc_scores else 0.5
    
    @staticmethod
    def _compute_uncertainty_calibration(predictions, labels, uncertainties, n_bins=10):
        calibration_errors = []
        
        for i in range(predictions.shape[1]):
            bin_boundaries = np.percentile(uncertainties[:, i], np.linspace(0, 100, n_bins + 1))
            
            for j in range(n_bins):
                mask = (uncertainties[:, i] >= bin_boundaries[j]) & (uncertainties[:, i] < bin_boundaries[j + 1])
                
                if np.sum(mask) > 0:
                    bin_accuracy = np.mean((predictions[mask, i] > 0.5) == labels[mask, i])
                    bin_uncertainty = np.mean(uncertainties[mask, i])
                    expected_error = bin_uncertainty
                    actual_error = 1 - bin_accuracy
                    
                    calibration_errors.append(np.abs(expected_error - actual_error))
        
        return np.mean(calibration_errors) if calibration_errors else 0.0
    
    @staticmethod
    def compute_calibration_metrics(predictions, labels, n_bins=15):
        metrics = {}
        
        pred_flat = predictions.flatten()
        label_flat = labels.flatten()
        
        metrics['ece'] = AdvancedMetricsCalculator._compute_ece(pred_flat, label_flat, n_bins)
        metrics['mce'] = AdvancedMetricsCalculator._compute_mce(pred_flat, label_flat, n_bins)
        metrics['ace'] = AdvancedMetricsCalculator._compute_ace(pred_flat, label_flat, n_bins)
        
        class_eces = []
        for i in range(predictions.shape[1]):
            class_ece = AdvancedMetricsCalculator._compute_ece(
                predictions[:, i], labels[:, i], n_bins
            )
            class_eces.append(class_ece)
        
        metrics['class_wise_ece'] = class_eces
        metrics['mean_class_ece'] = np.mean(class_eces)
        
        metrics['reliability_data'] = AdvancedMetricsCalculator._compute_reliability_data(
            pred_flat, label_flat, n_bins
        )
        
        return metrics
    
    @staticmethod
    def _compute_ece(predictions, labels, n_bins):
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        total_samples = len(predictions)
        
        for i in range(n_bins):
            mask = (predictions >= bin_boundaries[i]) & (predictions < bin_boundaries[i + 1])
            if i == n_bins - 1:
                mask = (predictions >= bin_boundaries[i]) & (predictions <= bin_boundaries[i + 1])
            
            if np.sum(mask) > 0:
                bin_accuracy = np.mean(labels[mask])
                bin_confidence = np.mean(predictions[mask])
                bin_weight = np.sum(mask) / total_samples
                ece += bin_weight * np.abs(bin_accuracy - bin_confidence)
        
        return ece
    
    @staticmethod
    def _compute_mce(predictions, labels, n_bins):
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        mce = 0.0
        
        for i in range(n_bins):
            mask = (predictions >= bin_boundaries[i]) & (predictions < bin_boundaries[i + 1])
            if i == n_bins - 1:
                mask = (predictions >= bin_boundaries[i]) & (predictions <= bin_boundaries[i + 1])
            
            if np.sum(mask) > 0:
                bin_accuracy = np.mean(labels[mask])
                bin_confidence = np.mean(predictions[mask])
                mce = max(mce, np.abs(bin_accuracy - bin_confidence))
        
        return mce
    
    @staticmethod
    def _compute_ace(predictions, labels, n_bins):
        sorted_predictions = np.sort(predictions)
        n_per_bin = len(predictions) // n_bins
        
        ace = 0.0
        for i in range(n_bins):
            start_idx = i * n_per_bin
            end_idx = (i + 1) * n_per_bin if i < n_bins - 1 else len(predictions)
            
            if end_idx > start_idx:
                bin_predictions = sorted_predictions[start_idx:end_idx]
                lower_bound = bin_predictions[0]
                upper_bound = bin_predictions[-1]
                
                mask = (predictions >= lower_bound) & (predictions <= upper_bound)
                
                if np.sum(mask) > 0:
                    bin_accuracy = np.mean(labels[mask])
                    bin_confidence = np.mean(predictions[mask])
                    bin_weight = np.sum(mask) / len(predictions)
                    ace += bin_weight * np.abs(bin_accuracy - bin_confidence)
        
        return ace
    
    @staticmethod
    def _compute_reliability_data(predictions, labels, n_bins):
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        accuracies = []
        confidences = []
        counts = []
        
        for i in range(n_bins):
            mask = (predictions >= bin_boundaries[i]) & (predictions < bin_boundaries[i + 1])
            if i == n_bins - 1:
                mask = (predictions >= bin_boundaries[i]) & (predictions <= bin_boundaries[i + 1])
            
            if np.sum(mask) > 0:
                accuracies.append(np.mean(labels[mask]))
                confidences.append(np.mean(predictions[mask]))
                counts.append(np.sum(mask))
            else:
                accuracies.append(0)
                confidences.append(0)
                counts.append(0)
        
        return {
            'accuracies': accuracies,
            'confidences': confidences,
            'counts': counts
        }