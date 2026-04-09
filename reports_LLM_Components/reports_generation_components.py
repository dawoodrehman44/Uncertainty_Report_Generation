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

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", FutureWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ================================
# ALL YOUR EXISTING BAYESIAN COMPONENTS (EXACTLY AS THEY ARE)
# ================================


# ================================
# NEW: HYBRID LLM INTEGRATION COMPONENTS
# ================================

class StructuredDiagnosticOutput:
    """Structured representation of your Bayesian model's output for LLM consumption"""
    def __init__(self, disease_logits, uncertainties, consistency_score):
        self.disease_names = [
            'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 
            'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 
            'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 
            'Pleural Other', 'Fracture', 'Support Devices'
        ]
        
        self.findings = self._process_findings(disease_logits, uncertainties)
        self.consistency = consistency_score
        self.uncertainties = uncertainties
        
    def _process_findings(self, logits, uncertainties):
        """Convert model outputs to structured findings"""
        probs = torch.sigmoid(logits)
        findings = []
        
        for i, disease in enumerate(self.disease_names):
            if disease == 'No Finding':
                continue
                
            finding = {
                'disease': disease,
                'probability': probs[i].item(),
                'epistemic_uncertainty': uncertainties['epistemic_uncertainty'][i].item(),
                'aleatoric_uncertainty': uncertainties['aleatoric_uncertainty'][i].item(),
                'total_uncertainty': uncertainties['total_uncertainty'][i].item(),
                'confidence_level': self._get_confidence_level(
                    probs[i].item(), 
                    uncertainties['total_uncertainty'][i].item()
                )
            }
            findings.append(finding)
        
        return sorted(findings, key=lambda x: x['probability'], reverse=True)
    
    def _get_confidence_level(self, prob, uncertainty):
        """Map probability and uncertainty to confidence level"""
        if prob > 0.7 and uncertainty < 0.3:
            return 'high'
        elif prob > 0.5 and uncertainty < 0.5:
            return 'moderate'
        elif prob > 0.3 and uncertainty < 0.7:
            return 'low'
        else:
            return 'very_low'
    
    def to_prompt(self):
        """Convert to structured prompt for LLM"""
        prompt_parts = []
        
        # Add consistency information
        if self.consistency < 0.5:
            prompt_parts.append("IMAGE_QUALITY: Low consistency detected (score: {:.2f})".format(
                self.consistency.item()))
        
        # Add findings
        prompt_parts.append("\nDIAGNOSTIC_FINDINGS:")
        for finding in self.findings:
            if finding['probability'] > 0.3:  # Only include relevant findings
                prompt_parts.append(
                    f"- {finding['disease']}: probability={finding['probability']:.2%}, "
                    f"epistemic={finding['epistemic_uncertainty']:.3f}, "
                    f"aleatoric={finding['aleatoric_uncertainty']:.3f}, "
                    f"confidence={finding['confidence_level']}"
                )
        
        # Add uncertainty summary
        mean_epistemic = np.mean([f['epistemic_uncertainty'] for f in self.findings])
        mean_aleatoric = np.mean([f['aleatoric_uncertainty'] for f in self.findings])
        
        prompt_parts.append(f"\nUNCERTAINTY_PROFILE:")
        prompt_parts.append(f"- Mean epistemic (model): {mean_epistemic:.3f}")
        prompt_parts.append(f"- Mean aleatoric (data): {mean_aleatoric:.3f}")
        
        return "\n".join(prompt_parts)

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    BitsAndBytesConfig
)

from typing import Optional, Dict, List
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import LoraConfig, get_peft_model, TaskType

class ImprovedHybridLLMReportGenerator(nn.Module):
    def __init__(self, feature_dim: int = 2048,
                 llm_model: str = "GanjinZero/biobart-base",
                 freeze_llm: bool = False):  # Removed use_lora parameter
        super().__init__()
        
        print(f"Report generator initialized with feature_dim={feature_dim}")
        
        # Adapter: feature_dim → 768
        self.feature_adapter = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 768),
            nn.LayerNorm(768),
            nn.GELU()
        )
        
        print(f"Loading medical LLM: {llm_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model)
        self.tokenizer.padding_side = 'left'
        
        # Add padding token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Add special tokens for medical reports
        special_tokens = {
            'additional_special_tokens': [
                '[FINDING]', '[IMPRESSION]', '[NORMAL]', '[ABNORMAL]',
                '[UNCERTAIN]', '[HIGH_CONF]', '[LOW_CONF]'
            ]
        }
        self.tokenizer.add_special_tokens(special_tokens)
        
        # Load model
        if "t5" in llm_model.lower() or "bart" in llm_model.lower():
            self.llm = AutoModelForSeq2SeqLM.from_pretrained(llm_model)
        else:
            self.llm = AutoModelForCausalLM.from_pretrained(llm_model)
        
        # Resize embeddings for special tokens
        self.llm.resize_token_embeddings(len(self.tokenizer))
        
        # CRITICAL FIX: Selective unfreezing for small datasets
        # Freeze encoder (keep biomedical knowledge from pre-training)
        if hasattr(self.llm, 'model') and hasattr(self.llm.model, 'encoder'):
            for param in self.llm.model.encoder.parameters():
                param.requires_grad = False
            print("Froze encoder (keeping biomedical knowledge)")
        
        # Unfreeze decoder completely (learn radiology report generation)
        if hasattr(self.llm, 'model') and hasattr(self.llm.model, 'decoder'):
            for param in self.llm.model.decoder.parameters():
                param.requires_grad = True
            print("Unfroze decoder (learning report generation)")
        
        # Unfreeze LM head (critical for text generation)
        if hasattr(self.llm, 'lm_head'):
            for param in self.llm.lm_head.parameters():
                param.requires_grad = True
            print("Unfroze LM head")
        
        # Print trainable stats
        trainable = sum(p.numel() for p in self.llm.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.llm.parameters())
        print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    
    def create_medical_prompt(self, structured_output):
        """Create prompts matching BioBart's pre-training distribution"""
        findings = [f for f in structured_output.findings if f['probability'] > 0.4][:5]
        
        if not findings:
            # For normal cases
            prompt = "Chest radiograph demonstrates: no acute cardiopulmonary abnormalities."
        else:
            # Build findings with uncertainty-aware language
            finding_strs = []
            for f in findings:
                disease = f['disease'].replace('_', ' ').lower()
                prob = f['probability']
                unc = f['total_uncertainty']
                
                # Modulate language based on uncertainty
                if prob > 0.7 and unc < 0.3:
                    prefix = "definite"
                elif prob > 0.5 and unc < 0.5:
                    prefix = "probable"
                else:
                    prefix = "possible"
                
                finding_strs.append(f"{prefix} {disease}")
            
            # Use PubMed-style language (BioBart's pre-training)
            prompt = f"Chest radiograph demonstrates: {', '.join(finding_strs)}."
        
        return prompt
    
    def forward(self, visual_features: torch.Tensor, 
                diagnostic_output: Dict,
                target_reports: Optional[List[str]] = None,
                training: bool = False):
        """
        Generate or train on reports
        
        Args:
            visual_features: Features from encoder [batch, feature_dim]
            diagnostic_output: Output from Bayesian framework
            target_reports: Ground truth reports for training
            training: Whether in training mode
        """
        batch_size = visual_features.size(0)
        
        # Adapt visual features to LLM embedding size
        adapted_features = self.feature_adapter(visual_features)  # [batch, 768]
        
        if training and target_reports is not None:
            # Training mode: compute loss against real reports
            loss = self._compute_report_loss(
                adapted_features, diagnostic_output, target_reports
            )
            return loss
        else:
            # Inference mode: generate reports
            reports = self._generate_reports(
                adapted_features, diagnostic_output
            )
            return reports
    
    def _compute_report_loss(self, features, diagnostic_output, target_reports):
        """Compute loss for report generation training - NO SCALING"""
        total_loss = 0
        batch_size = len(target_reports)
        valid_count = 0
        
        for i in range(batch_size):
            try:
                # Create structured representation
                structured = StructuredDiagnosticOutput(
                    diagnostic_output['disease_logits'][i],
                    {k: v[i] for k, v in diagnostic_output['class_uncertainties'].items()},
                    diagnostic_output['consistency_score'][i]
                )
                
                # Create medical prompt
                prompt = self.create_medical_prompt(structured)
                
                # Tokenize prompt and target
                inputs = self.tokenizer(
                    prompt, 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=128,
                    padding='max_length'
                )
                targets = self.tokenizer(
                    target_reports[i], 
                    return_tensors="pt",
                    truncation=True, 
                    max_length=256,
                    padding='max_length'
                )
                
                inputs = {k: v.to(self.llm.device) for k, v in inputs.items()}
                targets = {k: v.to(self.llm.device) for k, v in targets.items()}
                
                # Forward through LLM
                outputs = self.llm(**inputs, labels=targets['input_ids'])
                
                # CRITICAL FIX: Use RAW loss - no scaling
                sample_loss = outputs.loss
                
                total_loss += sample_loss
                valid_count += 1
                
            except Exception as e:
                continue
        
        if valid_count == 0:
            return torch.tensor(0.0, device=self.llm.device, requires_grad=True)
        
        return total_loss / valid_count
    
    def _generate_reports(self, features, diagnostic_output):
        """Generate reports during inference"""
        batch_size = features.size(0)
        reports = []
        
        for i in range(batch_size):
            structured = StructuredDiagnosticOutput(
                diagnostic_output['disease_logits'][i],
                {k: v[i] for k, v in diagnostic_output['class_uncertainties'].items()},
                diagnostic_output['consistency_score'][i]
            )
            
            prompt = self.create_medical_prompt(structured)
            
            # Tokenize and generate
            inputs = self.tokenizer(prompt, return_tensors="pt", 
                                   truncation=True, max_length=256)
            inputs = {k: v.to(self.llm.device) for k, v in inputs.items()}
            
            # Temperature based on uncertainty
            mean_uncertainty = structured.uncertainties['total_uncertainty'].mean().item()
            temperature = 0.7 + (mean_uncertainty * 0.3)  # Higher temp for higher uncertainty
            
            with torch.no_grad():
                outputs = self.llm.generate(
                    **inputs,
                    max_length=400,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9,
                    num_beams=3,
                    early_stopping=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            report = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            report = self._format_report(report, structured)
            reports.append(report)
        
        return reports
    
    def _format_report(self, report: str, structured_output):
        """Format report into standard radiology structure"""
        lines = report.split('. ')
        
        findings = []
        impression = []
        
        for line in lines:
            if any(disease in line.lower() for disease in 
                   ['cardiomegaly', 'opacity', 'effusion', 'pneumonia', 'consolidation']):
                findings.append(line)
            else:
                impression.append(line)
        
        # Build formatted report
        formatted = "CHEST X-RAY REPORT\n\n"
        formatted += "FINDINGS:\n"
        
        if findings:
            for f in findings:
                formatted += f"- {f.strip()}.\n"
        else:
            formatted += "- No acute cardiopulmonary findings.\n"
        
        formatted += "\nIMPRESSION:\n"
        if impression:
            formatted += ' '.join(impression) + '\n'
        else:
            if structured_output.findings[0]['probability'] > 0.7:
                formatted += f"- {structured_output.findings[0]['disease']} identified.\n"
            else:
                formatted += "- No acute findings. Clinical correlation recommended.\n"
        
        # Add uncertainty note if significant
        mean_epistemic = np.mean([f['epistemic_uncertainty'] for f in structured_output.findings])
        if mean_epistemic > 0.4:
            formatted += "\nNOTE: Image quality or positioning may limit evaluation. "
            formatted += "Consider additional imaging if clinically indicated.\n"
        
        return formatted
    
    def _post_process_report(self, report: str, structured_output: StructuredDiagnosticOutput):
        """Ensure report accurately reflects uncertainties"""
        # Add uncertainty metrics if not present
        if "UNCERTAINTY" not in report:
            uncertainty_section = self._generate_uncertainty_section(structured_output)
            report += f"\n\n{uncertainty_section}"
        
        return report
    
    def _generate_uncertainty_section(self, structured_output: StructuredDiagnosticOutput):
        """Generate uncertainty section for report"""
        lines = ["UNCERTAINTY ASSESSMENT:"]
        
        # Overall confidence
        high_conf = sum(1 for f in structured_output.findings if f['confidence_level'] == 'high')
        total = len([f for f in structured_output.findings if f['probability'] > 0.3])
        
        lines.append(f"- High confidence findings: {high_conf}/{total}")
        
        # Epistemic vs Aleatoric
        mean_epistemic = np.mean([f['epistemic_uncertainty'] for f in structured_output.findings])
        mean_aleatoric = np.mean([f['aleatoric_uncertainty'] for f in structured_output.findings])
        
        if mean_epistemic > mean_aleatoric:
            lines.append("- Model uncertainty elevated - additional views may be beneficial")
        elif mean_aleatoric > mean_epistemic * 1.5:
            lines.append("- Data uncertainty elevated - consider image quality factors")
        
        return "\n".join(lines)