import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import traceback
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

# Local project imports
from Metrics_Calculation.advance_metrics_calculator import AdvancedMetricsCalculator
from Metrics_Calculation.report_metrics import MedicalReportMetrics

# Optional (only if used in your code elsewhere)
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import logging
import warnings
import json
import math
from typing import Dict, List, Tuple, Optional

def validate_epoch_with_hybrid_llm(model, valid_loader, device, epoch, generate_llm_reports=False):
    """Modified validation with comprehensive report metrics"""
    model.eval()
    
    # Metric initialization
    all_preds = []
    all_labels = []
    all_epistemic = []
    all_aleatoric = []
    all_consistency = []
    all_total_uncertainty = []
    
    # Report collections
    all_generated_reports = []
    all_reference_reports = []
    
    metrics_calculator = AdvancedMetricsCalculator()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(valid_loader):
            if batch is None or any(x is None for x in batch):
                continue
            
            try:
                images, labels, reference_reports = batch
                images = images.to(device)
                labels = labels.to(device)

                # === MC Dropout Evaluation ===
                outputs_mc = model(
                    {'images': images}, device,
                    mc_dropout=True, n_mc=50
                )

                preds = torch.sigmoid(outputs_mc['disease_logits']).cpu().numpy()
                all_preds.append(preds)
                all_labels.append(labels.cpu().numpy())

                if 'epistemic_uncertainty' in outputs_mc:
                    all_epistemic.append(outputs_mc['epistemic_uncertainty'].cpu().numpy())

                # === Uncertainty Decomposition ===
                outputs_unc = model(
                    {'images': images}, device,
                    return_uncertainty_decomposition=True,
                    generate_report=False
                )

                if 'class_uncertainties' in outputs_unc:
                    all_aleatoric.append(
                        outputs_unc['class_uncertainties']['aleatoric_uncertainty'].cpu().numpy()
                    )
                    all_total_uncertainty.append(
                        outputs_unc['class_uncertainties']['total_uncertainty'].cpu().numpy()
                    )

                if 'consistency_score' in outputs_unc:
                    all_consistency.append(outputs_unc['consistency_score'].cpu().numpy())
                
                # === Collect reports ONCE (for both metrics and display) ===
                if generate_llm_reports and len(all_generated_reports) < 503:  # 500 metrics + 3 display
                    try:
                        outputs_report = model({'images': images}, device, generate_report=True)
                        
                        if 'llm_reports' in outputs_report:
                            for gen_report, ref_report in zip(outputs_report['llm_reports'], reference_reports):
                                if ref_report and isinstance(ref_report, str) and len(ref_report) > 10:
                                    all_generated_reports.append(gen_report)
                                    all_reference_reports.append(ref_report)
                                    
                                    if len(all_generated_reports) >= 503:
                                        break
                    except Exception as e:
                        print(f"Report collection error in batch {batch_idx}: {e}")

            except Exception as e:
                print(f"Validation error in batch {batch_idx}: {e}")
                continue
    
    # === Metrics Calculation ===
    if not all_preds:
        return None
    
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    metrics = {}

    # ROC-AUC
    auc_scores = []
    for i in range(all_preds.shape[1]):
        if len(np.unique(all_labels[:, i])) > 1:
            auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
            auc_scores.append(auc)
    metrics['roc_auc_macro'] = np.mean(auc_scores) if auc_scores else 0.5
    metrics['roc_auc_per_class'] = auc_scores

    # Average Precision
    ap_scores = []
    for i in range(all_preds.shape[1]):
        if len(np.unique(all_labels[:, i])) > 1:
            ap = average_precision_score(all_labels[:, i], all_preds[:, i])
            ap_scores.append(ap)
    metrics['average_precision'] = np.mean(ap_scores) if ap_scores else 0.0

    # Calibration metrics
    calibration_metrics = metrics_calculator.compute_calibration_metrics(all_preds, all_labels)
    metrics.update(calibration_metrics)

    # Uncertainty metrics
    if all_epistemic and all_aleatoric and all_total_uncertainty:
        all_epistemic = np.concatenate(all_epistemic)
        all_aleatoric = np.concatenate(all_aleatoric)
        all_total_uncertainty = np.concatenate(all_total_uncertainty)

        metrics['epistemic_uncertainty_mean'] = np.mean(all_epistemic)
        metrics['epistemic_uncertainty_std'] = np.std(all_epistemic)
        metrics['aleatoric_uncertainty_mean'] = np.mean(all_aleatoric)
        metrics['aleatoric_uncertainty_std'] = np.std(all_aleatoric)
        metrics['total_uncertainty_mean'] = np.mean(all_total_uncertainty)

        errors = np.abs(all_preds - all_labels)
        correlations = []
        for i in range(all_preds.shape[1]):
            if np.std(all_total_uncertainty[:, i]) > 0 and np.std(errors[:, i]) > 0:
                corr = np.corrcoef(all_total_uncertainty[:, i], errors[:, i])[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)

        metrics['uncertainty_error_correlation'] = np.mean(correlations) if correlations else 0.0

        eps_ratio = all_epistemic / (all_total_uncertainty + 1e-8)
        metrics['epistemic_ratio_mean'] = np.mean(eps_ratio)
        metrics['epistemic_ratio_std'] = np.std(eps_ratio)

    if all_consistency:
        all_consistency = np.concatenate(all_consistency)
        metrics['mean_consistency'] = np.mean(all_consistency)
        metrics['std_consistency'] = np.std(all_consistency)

    # === Extract samples for display and prepare for metrics ===
    sample_reports = []
    if len(all_generated_reports) >= 3:
        sample_reports = all_generated_reports[:3]  # First 3 for display
        # Trim to 500 for metrics calculation
        all_generated_reports = all_generated_reports[:500]
        all_reference_reports = all_reference_reports[:500]

    # === Report Metrics Calculation ===
    if all_generated_reports and len(all_generated_reports) > 0:
        print(f"\nEvaluating {len(all_generated_reports)} generated reports...")
        
        report_evaluator = MedicalReportMetrics()
        
        try:
            report_metrics = report_evaluator.compute_all_metrics(
                all_generated_reports,
                all_reference_reports
            )
            
            # Add to main metrics with 'report_' prefix
            for k, v in report_metrics.items():
                metrics[f'report_{k}'] = v

            # # === CRITICAL: Compute validation n-gram loss ===
            # try:
            #     # Get tokenizer from model
            #     if hasattr(model, 'report_generator') and hasattr(model.report_generator, 'tokenizer'):
            #         tokenizer = model.report_generator.tokenizer
                    
            #         val_ngram_loss = compute_validation_ngram_loss(
            #             all_generated_reports,
            #             all_reference_reports,
            #             tokenizer
            #         )
                    
            #         metrics['ngram_loss_component'] = val_ngram_loss
                    
            #         # Also get training losses if available
            #         if hasattr(model.report_generator, 'latest_losses'):
            #             metrics['ce_loss_component'] = model.report_generator.latest_losses.get('ce', 0.0)
                    
            #         print(f"\nLoss–Metric Correlation Check:")
            #         print(f"  N-gram Loss: {val_ngram_loss:.4f}")
            #         print(f"  BLEU-4: {metrics.get('report_bleu-4', 0):.4f}")
            #         print(f"  Expected: Lower n-gram loss should = higher BLEU")
                    
            # except Exception as e:
            #     print(f"N-gram loss computation error: {e}")
            #     import traceback
            #     traceback.print_exc()
                
        except Exception as e:
            print(f"Report metrics computation error: {e}")
            import traceback
            traceback.print_exc()

    # === Print Results ===
    print(f"\nValidation Results:")
    print(f"ROC-AUC (Macro): {metrics['roc_auc_macro']:.4f}")
    print(f"Average Precision: {metrics['average_precision']:.4f}")
    print(f"ECE: {metrics['ece']:.4f}")
    print(f"MCE: {metrics['mce']:.4f}")
    
    if 'uncertainty_error_correlation' in metrics:
        print(f"\nUncertainty Metrics:")
        print(f"  Uncertainty-Error Correlation: {metrics['uncertainty_error_correlation']:.4f}")
        
        if 'epistemic_ratio_mean' in metrics:
            print(f"  Epistemic Ratio: {metrics['epistemic_ratio_mean']:.3f} ± {metrics['epistemic_ratio_std']:.3f}")
        
        if 'epistemic_uncertainty_mean' in metrics:
            print(f"  Epistemic Uncertainty: {metrics['epistemic_uncertainty_mean']:.4f} ± {metrics['epistemic_uncertainty_std']:.4f}")
        
        if 'aleatoric_uncertainty_mean' in metrics:
            print(f"  Aleatoric Uncertainty: {metrics['aleatoric_uncertainty_mean']:.4f} ± {metrics['aleatoric_uncertainty_std']:.4f}")
    
    if 'mean_consistency' in metrics:
        print(f"\nConsistency Metrics:")
        print(f"  Mean Consistency: {metrics['mean_consistency']:.4f}")
        print(f"  Std Consistency: {metrics['std_consistency']:.4f}")
    
    # === Print Report Metrics ===
    if any(k.startswith('report_') for k in metrics.keys()):
        print(f"\n{'='*80}")
        print("REPORT GENERATION METRICS")
        print(f"{'='*80}")
        print(f"Evaluated on {len(all_generated_reports)} report pairs")
        
        if 'report_bleu-1' in metrics:
            print(f"\nLexical Overlap:")
            print(f"  BLEU-1: {metrics.get('report_bleu-1', 0):.4f}")
            print(f"  BLEU-4: {metrics.get('report_bleu-4', 0):.4f}")
            print(f"  ROUGE-L: {metrics.get('report_rouge-l', 0):.4f}")
            print(f"  METEOR: {metrics.get('report_meteor', 0):.4f}")
        
        if 'report_bertscore_f1' in metrics:
            print(f"\nSemantic Similarity (BiomedBERT):")
            print(f"  Precision: {metrics['report_bertscore_precision']:.4f}")
            print(f"  Recall: {metrics['report_bertscore_recall']:.4f}")
            print(f"  F1: {metrics['report_bertscore_f1']:.4f}")
        
        if 'report_clinical_f1' in metrics:
            print(f"\nClinical Accuracy:")
            print(f"  Precision: {metrics['report_clinical_precision']:.4f}")
            print(f"  Recall: {metrics['report_clinical_recall']:.4f}")
            print(f"  F1: {metrics['report_clinical_f1']:.4f}")
        
        if 'report_report_completeness' in metrics:
            print(f"\nRadiology-Specific:")
            print(f"  Report Completeness: {metrics['report_report_completeness']:.4f}")
            print(f"  Uncertainty Appropriateness: {metrics['report_uncertainty_appropriateness']:.4f}")
            print(f"  Length Similarity: {metrics['report_length_similarity']:.4f}")
        
        print(f"{'='*80}\n")
    
    # Print sample reports for inspection
    if sample_reports:
        print("\n===== Sample Validation Reports =====")
        for i, report in enumerate(sample_reports):
            print(f"\nValidation Report {i+1}:")
            print(report)
        print("======================================\n")

    return metrics