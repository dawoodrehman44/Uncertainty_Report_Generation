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
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Framework_Components.bayesian_componenets import EnhancedBayesianFramework
from Multi_Objective_loss.multi_objective_loss_Calculation import EnhancedMultiObjectiveLoss
from reports_LLM_Components.reports_generation_components import ImprovedHybridLLMReportGenerator

class EnhancedMultiAgentBayesianModel(nn.Module):
    def __init__(self, base_encoder, num_classes=14, hidden_dim=512, dropout_rate=0.3,
                 enable_llm_reports=False, llm_model=None):
        super().__init__()
        self.encoder = base_encoder
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.enable_llm_reports = enable_llm_reports
        
        # Detect encoder output dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 320, 320)
            dummy_batch = {'images': dummy_input}
            dummy_output = base_encoder(dummy_batch, 'cpu')
            
            if 'features' in dummy_output:
                visual_dim = dummy_output['features'].shape[-1]
                print(f"✓ Using visual features: {visual_dim} dimensions")
            else:
                visual_dim = dummy_output['cls_pred'].shape[-1]
                print(f"⚠️ WARNING: No visual features, using logits: {visual_dim} dimensions")
        
        # Bayesian framework
        self.bayesian_framework = EnhancedBayesianFramework(
            input_dim=hidden_dim,
            num_diseases=num_classes
        )
        
        self.loss_function = EnhancedMultiObjectiveLoss(num_classes)
        
        # FIX: Use visual_dim (2048) instead of num_classes (14)
        self.feature_projection = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),  # ← CHANGED: 2048 → 512
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        self.feature_enhancer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Report generator
        if enable_llm_reports:
            self.report_generator = ImprovedHybridLLMReportGenerator(
                feature_dim=visual_dim,
                llm_model=llm_model,
                freeze_llm=False
            )
        else:
            self.report_generator = None
    
    def forward(self, batch, device, mc_dropout=False, n_mc=10, 
                return_uncertainty_decomposition=False, generate_report=False,
                train_reports=False, target_reports=None):
        
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                 for k, v in batch.items()}
        
        encoder_output = self.encoder(batch, device)
        
        # FIX: Use visual features for Bayesian pathway
        if 'features' in encoder_output:
            visual_features = encoder_output['features']  # 2048-dim for Bayesian
        else:
            visual_features = encoder_output['cls_pred']  # Fallback
        
        # FIX: Project from visual features (2048), not logits (14)
        projected_features = self.feature_projection(visual_features)
        enhanced_features = self.feature_enhancer(projected_features)
        
        # MC dropout remains the same
        if mc_dropout:
            self.train()
            mc_outputs = []
            
            for _ in range(n_mc):
                output = self.bayesian_framework(enhanced_features, return_all_outputs=False)
                mc_outputs.append(torch.sigmoid(output['disease_logits']).detach())
            
            mc_preds = torch.stack(mc_outputs, dim=0)
            mean_pred = mc_preds.mean(dim=0)
            epistemic_uncertainty = mc_preds.var(dim=0)
            
            predictive_entropy = -torch.sum(mean_pred * torch.log(mean_pred + 1e-8), dim=-1)
            expected_entropy = -torch.mean(
                torch.sum(mc_preds * torch.log(mc_preds + 1e-8), dim=-1),
                dim=0
            )
            mutual_info = predictive_entropy - expected_entropy
            
            self.eval()
            
            return {
                'disease_logits': torch.logit(mean_pred.clamp(min=1e-8, max=1-1e-8)),
                'epistemic_uncertainty': epistemic_uncertainty,
                'predictive_entropy': predictive_entropy,
                'mutual_information': mutual_info,
                'mc_samples': mc_preds
            }
        
        # Standard forward
        else:
            outputs = self.bayesian_framework(
                enhanced_features, 
                return_all_outputs=return_uncertainty_decomposition
            )
            
            if return_uncertainty_decomposition:
                self._add_uncertainty_analysis(outputs)
            
            # Report training
            if train_reports and self.report_generator is not None and target_reports is not None:
                try:
                    report_loss = self.report_generator(
                        visual_features,  # Use same visual features
                        outputs,
                        target_reports=target_reports,
                        training=True
                    )
                    outputs['report_loss'] = report_loss
                except Exception as e:
                    print(f"Report training error: {e}")
                    outputs['report_loss'] = torch.tensor(0.0, device=device)
            
            # Report generation
            if generate_report and (not self.training) and self.report_generator is not None:
                try:
                    reports = self.report_generator(visual_features, outputs, training=False)
                    outputs['llm_reports'] = reports
                except Exception as e:
                    print(f"Report generation error: {e}")
            
            return outputs
    
    def _add_uncertainty_analysis(self, outputs):
        """Identical to no-reports version"""
        epistemic = outputs['class_uncertainties']['epistemic_uncertainty']
        aleatoric = outputs['class_uncertainties']['aleatoric_uncertainty']
        
        total_unc = epistemic + aleatoric + 1e-8
        outputs['uncertainty_ratios'] = {
            'epistemic_ratio': epistemic / total_unc,
            'aleatoric_ratio': aleatoric / total_unc
        }
        
        outputs['uncertainty_stats'] = {
            'epistemic_mean': epistemic.mean(dim=-1),
            'epistemic_std': epistemic.std(dim=-1),
            'aleatoric_mean': aleatoric.mean(dim=-1),
            'aleatoric_std': aleatoric.std(dim=-1),
            'total_mean': total_unc.mean(dim=-1),
            'total_std': total_unc.std(dim=-1)
        }
    
    def compute_loss(self, outputs, disease_labels, epoch=0):
        """Identical to no-reports version"""
        device = disease_labels.device
        targets = {'diseases': disease_labels}

        self.loss_function = self.loss_function.to(device)
        outputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                   for k, v in outputs.items()}
        targets = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                   for k, v in targets.items()}

        return self.loss_function(outputs, targets, epoch=epoch)