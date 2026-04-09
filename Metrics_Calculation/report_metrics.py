import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from evaluate import load

class MedicalReportMetrics:
    """Comprehensive metrics for radiology report evaluation"""
    
    def __init__(self):
        # NLG metrics with better error handling
        print("Initializing report metrics...")
        
        # BLEU
        try:
            self.bleu = load('bleu')
            print("✓ BLEU loaded successfully")
        except Exception as e:
            print(f"✗ BLEU load failed: {e}")
            self.bleu = None
        
        # ROUGE
        try:
            self.rouge = load('rouge')
            print("✓ ROUGE loaded successfully")
        except Exception as e:
            print(f"✗ ROUGE load failed: {e}")
            self.rouge = None
        
        # METEOR
        try:
            self.meteor = load('meteor')
            print("✓ METEOR loaded successfully")
        except Exception as e:
            print(f"✗ METEOR load failed: {e}")
            self.meteor = None
        
        self.clinical_labels = [
            'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 
            'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 
            'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 
            'Pleural Other', 'Fracture', 'Support Devices'
        ]
    
    def compute_all_metrics(self, generated_reports, reference_reports):
        """Compute all metrics for generated vs reference reports"""
        metrics = {}
        
        # Clean reports first
        generated_clean = [r[:1000] if r else "" for r in generated_reports]
        reference_clean = [r[:1000] if r else "" for r in reference_reports]
        
        # Filter out empty pairs
        valid_pairs = [(g, r) for g, r in zip(generated_clean, reference_clean) 
                    if g and r and len(g) > 10 and len(r) > 10]
        
        if not valid_pairs:
            print("No valid report pairs found!")
            return metrics
        
        generated_clean, reference_clean = zip(*valid_pairs)
        generated_clean = list(generated_clean)
        reference_clean = list(reference_clean)
        
        print(f"Processing {len(valid_pairs)} valid report pairs")
        
        # Debug: Check report lengths
        gen_lengths = [len(r.split()) for r in generated_clean]
        ref_lengths = [len(r.split()) for r in reference_clean]
        print(f"\nReport length statistics:")
        print(f"  Generated: min={min(gen_lengths)}, max={max(gen_lengths)}, mean={np.mean(gen_lengths):.1f}")
        print(f"  Reference: min={min(ref_lengths)}, max={max(ref_lengths)}, mean={np.mean(ref_lengths):.1f}")
        
        # 1. BLEU scores - COMPUTE REGARDLESS OF LOAD STATUS
        print("\nAttempting BLEU computation...")
        try:
            from evaluate import load
            if self.bleu is None:
                print("  BLEU wasn't loaded in __init__, loading now...")
                self.bleu = load('bleu')
            
            bleu_results = self.bleu.compute(
                predictions=generated_clean,
                references=[[r] for r in reference_clean],
                max_order=4
            )
            metrics['bleu-1'] = bleu_results['precisions'][0]
            metrics['bleu-2'] = bleu_results['precisions'][1]
            metrics['bleu-3'] = bleu_results['precisions'][2]
            metrics['bleu-4'] = bleu_results['bleu']
            print(f"✓ BLEU-1: {metrics['bleu-1']:.4f}")
            print(f"✓ BLEU-2: {metrics['bleu-2']:.4f}")
            print(f"✓ BLEU-3: {metrics['bleu-3']:.4f}")
            print(f"✓ BLEU-4: {metrics['bleu-4']:.4f}")
        except Exception as e:
            print(f"❌ BLEU computation failed: {e}")
            import traceback
            traceback.print_exc()
            metrics['bleu-1'] = 0.0
            metrics['bleu-2'] = 0.0
            metrics['bleu-3'] = 0.0
            metrics['bleu-4'] = 0.0

        # 2. ROUGE scores - COMPUTE REGARDLESS OF LOAD STATUS
        print("\nAttempting ROUGE computation...")
        try:
            from evaluate import load
            if self.rouge is None:
                print("  ROUGE wasn't loaded in __init__, loading now...")
                self.rouge = load('rouge')
            
            rouge_results = self.rouge.compute(
                predictions=generated_clean,
                references=reference_clean,
                use_stemmer=True
            )
            metrics['rouge-1'] = rouge_results['rouge1']
            metrics['rouge-2'] = rouge_results['rouge2']
            metrics['rouge-l'] = rouge_results['rougeL']
            print(f"✓ ROUGE-1: {metrics['rouge-1']:.4f}")
            print(f"✓ ROUGE-2: {metrics['rouge-2']:.4f}")
            print(f"✓ ROUGE-L: {metrics['rouge-l']:.4f}")
        except Exception as e:
            print(f"❌ ROUGE computation failed: {e}")
            import traceback
            traceback.print_exc()
            metrics['rouge-1'] = 0.0
            metrics['rouge-2'] = 0.0
            metrics['rouge-l'] = 0.0

        # 3. METEOR - COMPUTE REGARDLESS OF LOAD STATUS
        print("\nAttempting METEOR computation...")
        try:
            from evaluate import load
            if self.meteor is None:
                print("  METEOR wasn't loaded in __init__, loading now...")
                self.meteor = load('meteor')
            
            meteor_results = self.meteor.compute(
                predictions=generated_clean,
                references=reference_clean
            )
            metrics['meteor'] = meteor_results['meteor']
            print(f"✓ METEOR: {metrics['meteor']:.4f}")
        except Exception as e:
            print(f"❌ METEOR computation failed: {e}")
            import traceback
            traceback.print_exc()
            metrics['meteor'] = 0.0
        
        # 4. Clinical Efficacy Metrics
        try:
            clinical_metrics = self._compute_clinical_metrics(generated_clean, reference_clean)
            metrics.update(clinical_metrics)
            print(f"✓ Clinical metrics computed")
        except Exception as e:
            print(f"❌ Clinical metrics failed: {e}")
        
        # 5. Radiology-specific metrics
        try:
            radiology_metrics = self._compute_radiology_metrics(generated_clean, reference_clean)
            metrics.update(radiology_metrics)
            print(f"✓ Radiology metrics computed")
        except Exception as e:
            print(f"❌ Radiology metrics failed: {e}")
        
        return metrics
    
    def _compute_clinical_metrics(self, generated, reference):
        """Clinical efficacy metrics - extract findings and compare"""
        metrics = {}
        
        gen_entities = [self._extract_clinical_entities(r) for r in generated]
        ref_entities = [self._extract_clinical_entities(r) for r in reference]
        
        tp = fp = fn = 0
        
        for gen_ent, ref_ent in zip(gen_entities, ref_entities):
            gen_set = set(gen_ent)
            ref_set = set(ref_ent)
            
            tp += len(gen_set & ref_set)
            fp += len(gen_set - ref_set)
            fn += len(ref_set - gen_set)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics['clinical_precision'] = precision
        metrics['clinical_recall'] = recall
        metrics['clinical_f1'] = f1
        
        return metrics

    def _extract_clinical_entities(self, report):
        """Extract clinical findings from report"""
        entities = []
        report_lower = report.lower()
        
        for label in self.clinical_labels:
            label_lower = label.lower()
            if label_lower in report_lower:
                entities.append(label_lower)
        
        negation_phrases = ['no ', 'without ', 'negative for ', 'absence of ']
        for phrase in negation_phrases:
            if phrase in report_lower:
                idx = report_lower.find(phrase)
                context = report_lower[idx:idx+50]
                for label in self.clinical_labels:
                    if label.lower() in context:
                        entities.append(f'no_{label.lower()}')
        
        return entities
    
    def _compute_radiology_metrics(self, generated, reference):
        """Radiology-specific metrics"""
        metrics = {}
        
        # Report completeness
        required_sections = ['findings', 'impression']
        completeness_scores = []
        
        for report in generated:
            report_lower = report.lower()
            score = sum(1 for section in required_sections if section in report_lower)
            completeness_scores.append(score / len(required_sections))
        
        metrics['report_completeness'] = np.mean(completeness_scores)
        
        # Uncertainty expression appropriateness
        uncertainty_terms = ['possible', 'probable', 'likely', 'may', 'could', 'cannot exclude']
        uncertainty_scores = []
        
        for gen, ref in zip(generated, reference):
            gen_has_uncertainty = any(term in gen.lower() for term in uncertainty_terms)
            ref_has_uncertainty = any(term in ref.lower() for term in uncertainty_terms)
            
            if gen_has_uncertainty == ref_has_uncertainty:
                uncertainty_scores.append(1.0)
            else:
                uncertainty_scores.append(0.5)
        
        metrics['uncertainty_appropriateness'] = np.mean(uncertainty_scores)
        
        # Report length similarity
        len_ratios = []
        for gen, ref in zip(generated, reference):
            ratio = min(len(gen), len(ref)) / max(len(gen), len(ref))
            len_ratios.append(ratio)
        
        metrics['length_similarity'] = np.mean(len_ratios)
        
        return metrics