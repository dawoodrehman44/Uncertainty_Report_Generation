import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import roc_auc_score, average_precision_score
import warnings
import os
import json
import math
from typing import Dict, List, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import logging

class VariationalLinear(nn.Module):
    """EXACTLY YOUR EXISTING VariationalLinear - NO CHANGES"""
    def __init__(self, in_features, out_features, prior_std=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight_mean = nn.Parameter(torch.randn(out_features, in_features) * np.sqrt(1.0 / in_features))
        self.weight_logvar = nn.Parameter(torch.ones(out_features, in_features) * -8.0)
        
        self.bias_mean = nn.Parameter(torch.zeros(out_features))
        self.bias_logvar = nn.Parameter(torch.ones(out_features) * -8.0)
        
        self.register_buffer('prior_std', torch.tensor(prior_std))
        
    def forward(self, x, sample_posterior=True):
        device = x.device
        epoch = getattr(self, '_current_epoch', 0)
        
        if self.training and sample_posterior and epoch:
            weight_std = torch.exp(0.5 * self.weight_logvar)
            weight = self.weight_mean + weight_std * torch.randn_like(self.weight_mean)
            
            bias_std = torch.exp(0.5 * self.bias_logvar)
            bias = self.bias_mean + bias_std * torch.randn_like(self.bias_mean)
        else:
            weight = self.weight_mean
            bias = self.bias_mean
        
        output = F.linear(x, weight, bias)
        kl_div = self.compute_kl_divergence()
        
        return output, kl_div
    
    def compute_kl_divergence(self):
        kl_weight = 0.5 * torch.sum(
            torch.exp(self.weight_logvar) + self.weight_mean.pow(2) - 1 - self.weight_logvar
        )
        kl_bias = 0.5 * torch.sum(
            torch.exp(self.bias_logvar) + self.bias_mean.pow(2) - 1 - self.bias_logvar
        )
        
        n_params = self.in_features * self.out_features + self.out_features
        return (kl_weight + kl_bias) / n_params
    
class HierarchicalBayesianEncoder(nn.Module):
    """EXACTLY YOUR EXISTING HierarchicalBayesianEncoder - NO CHANGES"""
    def __init__(self, input_dim, num_hierarchy_levels=3, dropout_rate=0.1):
        super().__init__()
        self.hierarchy_levels = num_hierarchy_levels
        
        dims = [input_dim // (2**i) for i in range(num_hierarchy_levels + 1)]
        
        self.bayesian_layers = nn.ModuleList([
            VariationalLinear(dims[i], dims[i+1])
            for i in range(num_hierarchy_levels)
        ])
        
        self.residual_projections = nn.ModuleList([
            nn.Linear(dims[i], dims[i+1]) if dims[i] != dims[i+1] else nn.Identity()
            for i in range(num_hierarchy_levels)
        ])
        
        self.attention_weights = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dims[i+1], 1),
                nn.Sigmoid()
            ) for i in range(num_hierarchy_levels)
        ])
        
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.ModuleList([
            nn.LayerNorm(dims[i+1]) for i in range(num_hierarchy_levels)
        ])
        
    def forward(self, x):
        hierarchical_features = []
        hierarchical_kl = []
        attention_scores = []
        
        current_features = x
        for i, (layer, residual, attention, norm) in enumerate(
            zip(self.bayesian_layers, self.residual_projections, 
                self.attention_weights, self.layer_norm)):
            
            sample_flag = self.training
            features, kl_div = layer(current_features, sample_posterior=sample_flag)
            features = self.activation(features)
            
            residual_features = residual(current_features)
            features = features + residual_features
            
            features = norm(features)
            features = self.dropout(features)
            
            att_score = attention(features)
            attention_scores.append(att_score)
            
            hierarchical_features.append(features)
            hierarchical_kl.append(kl_div)
            
            current_features = features
            
        aggregated_features = torch.zeros_like(hierarchical_features[-1])
        total_attention = sum(attention_scores)
        
        for feat, att in zip(hierarchical_features, attention_scores):
            if feat.shape == aggregated_features.shape:
                aggregated_features += feat * (att / (total_attention + 1e-8))
            
        return {
            'features': hierarchical_features,
            'kl_divergences': hierarchical_kl,
            'final_features': current_features,
            'aggregated_features': aggregated_features,
            'attention_scores': attention_scores
        }


class BayesianDiseaseClassificationAgent(nn.Module):
    """EXACTLY YOUR EXISTING BayesianDiseaseClassificationAgent - NO CHANGES"""
    def __init__(self, input_dim, num_diseases=14, num_mc_samples=10):
        super().__init__()
        self.num_diseases = num_diseases
        self.num_mc_samples = num_mc_samples
        
        hidden_dim = input_dim // 2
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_diseases)
        )
        
        self.epistemic_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_diseases),
            nn.Softplus()
        )
        
        self.aleatoric_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_diseases),
            nn.Softplus()
        )
        
    def forward(self, features, return_distribution=False):
        class_logits = self.classifier(features)
        
        if not self.training:
            self.eval()
            with torch.no_grad():
                for module in self.classifier.modules():
                    if isinstance(module, nn.Dropout):
                        module.train()
                
                mc_predictions = []
                for _ in range(20):
                    mc_predictions.append(self.classifier(features))
                
                mc_predictions = torch.stack(mc_predictions)
                epistemic_uncertainty = torch.var(torch.sigmoid(mc_predictions), dim=0)
        else:
            epistemic_uncertainty = self.epistemic_net(features)
        
        aleatoric_uncertainty = self.aleatoric_net(features)
        
        return {
            'logits': class_logits,
            'epistemic_uncertainty': epistemic_uncertainty,
            'aleatoric_uncertainty': aleatoric_uncertainty,
            'total_uncertainty': epistemic_uncertainty + aleatoric_uncertainty
        }


class EnhancedBayesianConsistencyAgent(nn.Module):
    """EXACTLY YOUR EXISTING EnhancedBayesianConsistencyAgent - NO CHANGES"""
    def __init__(self, input_dim, num_diseases=14):
        super().__init__()
        self.num_diseases = num_diseases
        
        self.feature_consistency = nn.Sequential(
            nn.Linear(input_dim + num_diseases, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.uncertainty_consistency = nn.Sequential(
            nn.Linear(num_diseases * 2, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, features, predictions, uncertainties):
        pred_probs = torch.sigmoid(predictions)
        combined_features = torch.cat([features, pred_probs], dim=-1)
        feat_consistency = self.feature_consistency(combined_features)
        
        total_uncertainty = torch.cat([
            uncertainties['epistemic_uncertainty'],
            uncertainties['aleatoric_uncertainty']
        ], dim=-1)
        uncertainty_consistency = self.uncertainty_consistency(total_uncertainty)
        
        total_consistency = 0.6 * feat_consistency + 0.4 * uncertainty_consistency
        
        return {
            'consistency_score': total_consistency,
            'feature_consistency': feat_consistency,
            'uncertainty_consistency': uncertainty_consistency
        }


class SimpleCalibration(nn.Module):
    """EXACTLY YOUR EXISTING SimpleCalibration - NO CHANGES"""
    def __init__(self, num_diseases=14):
        super().__init__()
        
        self.platt_scale = nn.Parameter(torch.ones(num_diseases))
        self.platt_bias = nn.Parameter(torch.zeros(num_diseases))
        self.temperature = nn.Parameter(torch.ones(num_diseases) * 1.5)
        
    def forward(self, logits, method='temperature'):
        if method == 'platt':
            return logits * self.platt_scale + self.platt_bias
        elif method == 'temperature':
            return logits / (self.temperature + 1e-8)
        else:
            temp_scaled = logits / (self.temperature + 1e-8)
            return temp_scaled * self.platt_scale + self.platt_bias


# [Keep ALL your existing loss classes - EnhancedMultiObjectiveLoss, etc. - EXACTLY as they are]

class EnhancedBayesianFramework(nn.Module):
    """EXACTLY YOUR EXISTING EnhancedBayesianFramework - NO CHANGES"""
    def __init__(self, input_dim, num_diseases=14):
        super().__init__()
        
        self.bayesian_encoder = HierarchicalBayesianEncoder(input_dim, num_hierarchy_levels=3)
        final_dim = input_dim // 8
        
        self.classification_agent = BayesianDiseaseClassificationAgent(final_dim, num_diseases)
        self.consistency_agent = EnhancedBayesianConsistencyAgent(final_dim, num_diseases)
        self.calibration = SimpleCalibration(num_diseases)
        
    def forward(self, features, return_all_outputs=False):
        encoded = self.bayesian_encoder(features)
        final_features = encoded['aggregated_features']
        
        class_output = self.classification_agent(final_features)
        
        consistency_output = self.consistency_agent(
            final_features,
            class_output['logits'],
            class_output
        )
        
        calibrated_logits = self.calibration(class_output['logits'], method='temperature')
        
        kl_divergences = encoded['kl_divergences']
        
        outputs = {
            'disease_logits': calibrated_logits,
            'raw_logits': class_output['logits'],
            'class_uncertainties': {
                'epistemic_uncertainty': class_output['epistemic_uncertainty'],
                'aleatoric_uncertainty': class_output['aleatoric_uncertainty'],
                'total_uncertainty': class_output['total_uncertainty']
            },
            'consistency_score': consistency_output['consistency_score'],
            'feature_consistency': consistency_output['feature_consistency'],
            'uncertainty_consistency': consistency_output['uncertainty_consistency'],
            'kl_divergences': kl_divergences
        }
        
        return outputs

