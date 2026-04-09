import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch import nn
import torch.nn.functional as F
class EnhancedMultiObjectiveLoss(nn.Module):
    """EXACTLY YOUR EXISTING EnhancedMultiObjectiveLoss - NO CHANGES"""
    def __init__(self, num_diseases=14):
        super().__init__()
        self.num_diseases = num_diseases
        
        self.classification_weight = 1.0
        self.uncertainty_weight = 0.1      
        self.calibration_weight = 0.1      
        self.consistency_weight = 0.02     
        self.kl_weight = 1e-3
        
        self.focal_alpha = 0.25
        self.focal_gamma = 2.0

    def _compute_ece_loss(self, probs, labels, n_bins=10):
        bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=probs.device)
        ece = torch.tensor(0.0, device=probs.device)
        
        for i in range(n_bins):
            mask = (probs >= bin_boundaries[i]) & (probs < bin_boundaries[i + 1])
            if i == n_bins - 1:
                mask = (probs >= bin_boundaries[i]) & (probs <= bin_boundaries[i + 1])
            
            if mask.sum() > 10:
                bin_accuracy = (probs[mask].round() == labels[mask]).float().mean()
                bin_confidence = probs[mask].mean()
                bin_weight = mask.sum().float() / probs.numel()
                ece += bin_weight * torch.abs(bin_accuracy - bin_confidence)
        
        return ece
        
    def forward(self, outputs, targets, epoch=0):
        losses = {}
        device = outputs['disease_logits'].device
        
        class_logits = outputs['disease_logits']
        disease_labels = targets['diseases'].float()
        
        epoch_factor = min(1.0, epoch / 20)
        
        probs = torch.sigmoid(class_logits)
        
        base_ce_loss = F.binary_cross_entropy_with_logits(
            class_logits, disease_labels, reduction='mean'
        )
        
        if epoch > 10:
            ce_loss_unreduced = F.binary_cross_entropy_with_logits(
                class_logits, disease_labels, reduction='none'
            )
            p_t = probs * disease_labels + (1 - probs) * (1 - disease_labels)
            focal_weight = self.focal_alpha * (1 - p_t) ** self.focal_gamma
            focal_loss = (focal_weight * ce_loss_unreduced).mean()
            
            classification_loss = 0.7 * base_ce_loss + 0.3 * focal_loss
        else:
            classification_loss = base_ce_loss
        
        if epoch > 5:
            smoothing_factor = min(0.1, epoch * 0.01)
            smooth_labels = disease_labels * (1 - smoothing_factor) + smoothing_factor / 2
            smooth_loss = F.binary_cross_entropy_with_logits(class_logits, smooth_labels)
            classification_loss = 0.9 * classification_loss + 0.1 * smooth_loss
        
        losses['classification'] = classification_loss * self.classification_weight
        
        if 'class_uncertainties' in outputs:
            uncertainty = outputs['class_uncertainties']['total_uncertainty']
            epistemic = outputs['class_uncertainties']['epistemic_uncertainty']
            aleatoric = outputs['class_uncertainties']['aleatoric_uncertainty']
            
            with torch.no_grad():
                pred_errors = torch.abs(probs - disease_labels)
            
            uncertainty_corr_loss = -torch.mean(uncertainty * pred_errors)
            
            confidence = torch.abs(probs - 0.5) * 2
            target_uncertainty = 0.2 * (1 - confidence) + 0.1
            uncertainty_mse = F.mse_loss(uncertainty, target_uncertainty.detach())
            
            epistemic_weight = max(0.3, 0.7 - epoch * 0.01)
            aleatoric_weight = 1 - epistemic_weight
            
            epistemic_target = target_uncertainty * epistemic_weight
            aleatoric_target = target_uncertainty * aleatoric_weight
            balance_loss = (F.mse_loss(epistemic, epistemic_target.detach()) + 
                        F.mse_loss(aleatoric, aleatoric_target.detach()))
            
            total_uncertainty_loss = (
                0.5 * uncertainty_corr_loss +
                0.3 * uncertainty_mse +
                0.2 * balance_loss
            )
            
            uncertainty_weight = self.uncertainty_weight * epoch_factor
            losses['uncertainty'] = total_uncertainty_loss * uncertainty_weight
        else:
            losses['uncertainty'] = torch.tensor(0.0, device=device)
        
        if epoch > 5:
            calibration_loss = self._compute_ece_loss(probs, disease_labels)
            
            brier_loss = torch.mean((probs - disease_labels) ** 2)
            
            confidence_penalty = torch.mean(
                torch.where(
                    (probs > 0.8) & (disease_labels < 0.5),
                    probs - 0.8,
                    torch.zeros_like(probs)
                ) + torch.where(
                    (probs < 0.2) & (disease_labels > 0.5),
                    0.2 - probs,
                    torch.zeros_like(probs)
                )
            )
            
            total_calibration_loss = (calibration_loss + 
                                    0.2 * brier_loss + 
                                    0.1 * confidence_penalty)
            
            calibration_weight = self.calibration_weight * min(1.0, (epoch - 5) / 15)
            losses['calibration'] = total_calibration_loss * calibration_weight
        else:
            losses['calibration'] = torch.tensor(0.0, device=device)
        
        if 'consistency_score' in outputs:
            consistency_target = min(0.9, 0.5 + epoch * 0.005)
            consistency_target = torch.ones_like(outputs['consistency_score']) * consistency_target
            
            consistency_loss = F.mse_loss(outputs['consistency_score'], consistency_target.detach())
            
            consistency_weight = self.consistency_weight * epoch_factor
            losses['consistency'] = consistency_loss * consistency_weight
        else:
            losses['consistency'] = torch.tensor(0.0, device=device)
        
        if 'kl_divergences' in outputs:
            if isinstance(outputs['kl_divergences'], list):
                total_kl = sum(outputs['kl_divergences'])
            else:
                total_kl = outputs['kl_divergences']
            
            batch_size = class_logits.size(0)
            normalized_kl = total_kl / batch_size
            
            kl_annealing = min(1.0, epoch / 30)
            
            losses['kl_divergence'] = normalized_kl * (self.kl_weight * kl_annealing)
        else:
            losses['kl_divergence'] = torch.tensor(0.0, device=device)
        
        if epoch > 10 and 'disease_logits' in outputs:
            pred_mean = torch.mean(probs, dim=0)
            pred_std = torch.std(probs, dim=0)
            
            diversity_loss = -torch.mean(pred_std)
            
            balance_loss = torch.mean((pred_mean - 0.5) ** 2)
            
            losses['diversity'] = (diversity_loss * 0.01 + balance_loss * 0.01) * epoch_factor
        
        total_loss = sum(losses.values())
        losses['total'] = total_loss
        
        return total_loss, {k: v.item() if torch.is_tensor(v) else v for k, v in losses.items()}