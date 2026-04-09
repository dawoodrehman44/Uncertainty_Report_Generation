"""
Multi-Objective Loss Functions for Uncertainty-Aware Report Generation
FINAL CORRECT VERSION - Use SimplifiedReportGenerationLoss (without fluency)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List
import numpy as np


# ============================================
# COMPONENT 1: Uncertainty Calibration Loss
# ============================================

class UncertaintyCalibrationLoss(nn.Module):
    """
    Ensure generated text confidence aligns with Bayesian uncertainties
    
    Concept: High epistemic uncertainty → use hedging language
             High aleatoric uncertainty → acknowledge data limitations
             High consistency → use confident language
    """
    
    def __init__(self):
        super().__init__()
        
        # Keywords that indicate confidence levels
        self.high_confidence_words = [
            'shows', 'demonstrates', 'reveals', 'confirmed', 'clear', 'evident'
        ]
        self.low_confidence_words = [
            'possible', 'likely', 'suggests', 'may', 'consistent with', 'cannot exclude'
        ]
        self.uncertainty_words = [
            'uncertain', 'unclear', 'limited', 'suboptimal', 'questionable'
        ]
    
    def forward(self, 
                generated_logits: torch.Tensor,
                generated_tokens: torch.Tensor,
                diagnostic_output: Dict,
                vocab) -> torch.Tensor:
        """
        Args:
            generated_logits: [batch, seq_len, vocab_size]
            generated_tokens: [batch, seq_len] - actual generated tokens
            diagnostic_output: Dict with uncertainties
            vocab: MedicalVocabulary instance
        
        Returns:
            calibration_loss: Scalar tensor
        """
        batch_size = generated_logits.size(0)
        device = generated_logits.device
        
        # Get uncertainty metrics
        epistemic = diagnostic_output['class_uncertainties']['epistemic_uncertainty']
        aleatoric = diagnostic_output['class_uncertainties']['aleatoric_uncertainty']
        consistency = diagnostic_output['consistency_score']
        
        # Compute average uncertainty per sample
        avg_epistemic = epistemic.mean(dim=1)  # [batch]
        avg_aleatoric = aleatoric.mean(dim=1)
        
        # Get token-level confidence from logits
        token_probs = F.softmax(generated_logits, dim=-1)
        token_confidence = token_probs.max(dim=-1)[0]  # [batch, seq_len]
        avg_token_confidence = token_confidence.mean(dim=1)  # [batch]
        
        # Expected behavior:
        # - High uncertainty → low token confidence
        # - Low uncertainty → high token confidence
        
        total_uncertainty = avg_epistemic + avg_aleatoric
        normalized_uncertainty = total_uncertainty / (total_uncertainty.max() + 1e-8)
        
        # Target token confidence should be inversely related to uncertainty
        target_confidence = 1.0 - normalized_uncertainty
        
        # MSE between actual and target confidence
        calibration_loss = F.mse_loss(avg_token_confidence, target_confidence)
        
        # Additional penalty: check for keyword misalignment
        keyword_penalty = self._compute_keyword_penalty(
            generated_tokens, total_uncertainty, consistency, vocab
        )
        
        return calibration_loss + 0.1 * keyword_penalty
    
    def _compute_keyword_penalty(self, 
                                  tokens: torch.Tensor,
                                  uncertainty: torch.Tensor,
                                  consistency: torch.Tensor,
                                  vocab) -> torch.Tensor:
        """
        Penalize using confident language when uncertain, and vice versa
        
        Args:
            tokens: [batch, seq_len]
            uncertainty: [batch] - total uncertainty
            consistency: [batch]
            vocab: MedicalVocabulary
        """
        batch_size = tokens.size(0)
        device = tokens.device
        penalties = []
        
        # Get keyword indices
        high_conf_ids = [vocab.word2idx.get(word, -1) for word in self.high_confidence_words]
        low_conf_ids = [vocab.word2idx.get(word, -1) for word in self.low_confidence_words]
        uncertainty_ids = [vocab.word2idx.get(word, -1) for word in self.uncertainty_words]
        
        high_conf_ids = [i for i in high_conf_ids if i != -1]
        low_conf_ids = [i for i in low_conf_ids if i != -1]
        uncertainty_ids = [i for i in uncertainty_ids if i != -1]
        
        for i in range(batch_size):
            sample_tokens = tokens[i].cpu().tolist()
            
            # Count keyword occurrences
            high_conf_count = sum(1 for t in sample_tokens if t in high_conf_ids)
            low_conf_count = sum(1 for t in sample_tokens if t in low_conf_ids)
            uncertain_count = sum(1 for t in sample_tokens if t in uncertainty_ids)
            
            # Penalize misalignment
            penalty = 0.0
            
            # High uncertainty but confident language → bad
            if uncertainty[i] > 0.6 and high_conf_count > low_conf_count:
                penalty += 0.5
            
            # Low uncertainty but hedging language → bad
            if uncertainty[i] < 0.3 and low_conf_count > high_conf_count:
                penalty += 0.3
            
            # Low consistency but no uncertainty keywords → bad
            if consistency[i] < 0.5 and uncertain_count == 0:
                penalty += 0.4
            
            penalties.append(penalty)
        
        return torch.tensor(penalties, device=device).mean()


# ============================================
# COMPONENT 2: Diversity Loss
# ============================================

class DiversityLoss(nn.Module):
    """
    Aggressive diversity enforcement to prevent mode collapse
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, 
                generated_tokens: torch.Tensor,
                generated_logits: torch.Tensor,
                vocab_size: int) -> Dict[str, torch.Tensor]:
        """
        Returns dict with multiple diversity metrics
        """
        batch_size, seq_len = generated_tokens.size()
        device = generated_tokens.device
        
        losses = {}
        
        # ===== 1. N-gram Diversity (ENHANCED) =====
        distinct_unigrams = set()
        distinct_bigrams = set()
        distinct_trigrams = set()
        total_unigrams = 0
        total_bigrams = 0
        total_trigrams = 0
        
        for i in range(batch_size):
            tokens = generated_tokens[i].cpu().tolist()
            
            # Unigrams
            distinct_unigrams.update(tokens)
            total_unigrams += len(tokens)
            
            # Bigrams
            for j in range(len(tokens) - 1):
                distinct_bigrams.add((tokens[j], tokens[j+1]))
                total_bigrams += 1
            
            # Trigrams
            for j in range(len(tokens) - 2):
                distinct_trigrams.add((tokens[j], tokens[j+1], tokens[j+2]))
                total_trigrams += 1
        
        distinct_1 = len(distinct_unigrams) / max(total_unigrams, 1)
        distinct_2 = len(distinct_bigrams) / max(total_bigrams, 1)
        distinct_3 = len(distinct_trigrams) / max(total_trigrams, 1)
        
        # Penalize LOW diversity (higher penalty)
        ngram_penalty = (1.0 - distinct_1) + (1.0 - distinct_2) + (1.0 - distinct_3)
        losses['ngram_diversity'] = ngram_penalty / 3.0
        
        # ===== 2. Entropy of Token Distribution =====
        token_counts = torch.bincount(
            generated_tokens.flatten(), 
            minlength=vocab_size
        ).float()
        token_probs = token_counts / (token_counts.sum() + 1e-8)
        token_probs = token_probs[token_probs > 0]
        
        entropy = -(token_probs * torch.log(token_probs + 1e-8)).sum()
        max_entropy = np.log(vocab_size)
        normalized_entropy = entropy / max_entropy
        
        # Penalize LOW entropy
        losses['token_entropy'] = 1.0 - normalized_entropy
        
        # ===== 3. Self-BLEU (Diversity Across Samples) =====
        # Lower self-BLEU = more diverse across samples
        if batch_size > 1:
            self_bleu_scores = []
            
            for i in range(min(batch_size, 10)):  # Sample 10 to save time
                ref_tokens = generated_tokens[i].cpu().tolist()
                
                for j in range(batch_size):
                    if i == j:
                        continue
                    
                    hyp_tokens = generated_tokens[j].cpu().tolist()
                    
                    # Compute unigram overlap
                    ref_set = set(ref_tokens)
                    hyp_set = set(hyp_tokens)
                    overlap = len(ref_set & hyp_set) / max(len(ref_set), 1)
                    
                    self_bleu_scores.append(overlap)
            
            if self_bleu_scores:
                self_bleu = np.mean(self_bleu_scores)
                losses['self_bleu'] = torch.tensor(self_bleu, device=device)
            else:
                losses['self_bleu'] = torch.tensor(0.0, device=device)
        else:
            losses['self_bleu'] = torch.tensor(0.0, device=device)
        
        # ===== 4. Repetition Penalty (ENHANCED) =====
        repetition_penalties = []
        
        for i in range(batch_size):
            tokens = generated_tokens[i].cpu().tolist()
            
            # Count repeated 3-grams
            trigrams = []
            for j in range(len(tokens) - 2):
                trigrams.append(tuple(tokens[j:j+3]))
            
            if len(trigrams) > 0:
                unique_trigrams = len(set(trigrams))
                repetition_ratio = 1.0 - (unique_trigrams / len(trigrams))
                repetition_penalties.append(repetition_ratio)
        
        if repetition_penalties:
            losses['repetition'] = torch.tensor(
                np.mean(repetition_penalties), 
                device=device
            )
        else:
            losses['repetition'] = torch.tensor(0.0, device=device)
        
        # ===== 5. Confidence Penalty (NEW!) =====
        # Penalize overconfident predictions (sign of mode collapse)
        token_probs_from_logits = F.softmax(generated_logits, dim=-1)
        max_probs = token_probs_from_logits.max(dim=-1)[0]  # [batch, seq_len]
        
        # If average confidence > 0.9, model is too confident (mode collapse)
        avg_confidence = max_probs.mean()
        confidence_penalty = torch.relu(avg_confidence - 0.85) * 2.0
        
        losses['overconfidence'] = confidence_penalty
        
        # ===== TOTAL DIVERSITY LOSS =====
        total_diversity_loss = (
            losses['ngram_diversity'] * 0.25 +
            losses['token_entropy'] * 0.20 +
            losses['self_bleu'] * 0.20 +
            losses['repetition'] * 0.20 +
            losses['overconfidence'] * 0.15
        )
        
        losses['total'] = total_diversity_loss
        
        return losses
    
class RepetitionPenaltyLoss(nn.Module):
    """
    Penalize repeated n-grams in generated reports
    """
    def __init__(self, ngram_size=3):
        super().__init__()
        self.ngram_size = ngram_size
    
    def forward(self, generated_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            generated_tokens: [batch, seq_len]
        Returns:
            penalty: scalar (0 to 1, higher = more repetition)
        """
        batch_size, seq_len = generated_tokens.size()
        
        total_penalty = 0.0
        
        for i in range(batch_size):
            tokens = generated_tokens[i].cpu().tolist()
            
            # Extract n-grams
            ngrams = []
            for j in range(len(tokens) - self.ngram_size + 1):
                ngram = tuple(tokens[j:j+self.ngram_size])
                ngrams.append(ngram)
            
            if len(ngrams) == 0:
                continue
            
            # Count repetitions
            unique_ngrams = len(set(ngrams))
            total_ngrams = len(ngrams)
            
            # Penalty increases with repetition
            # If all unique: penalty = 0
            # If all same: penalty = 1
            repetition_ratio = 1.0 - (unique_ngrams / total_ngrams)
            total_penalty += repetition_ratio
        
        return torch.tensor(total_penalty / batch_size, device=generated_tokens.device)

class GrammarFluencyLoss(nn.Module):
    """
    Use pre-trained LM to enforce grammatical fluency
    Based on: R2GenCMN, Show-Attend-Tell papers
    """
    def __init__(self):
        super().__init__()
        # Use small medical LM or BioBERT
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        
        self.lm = GPT2LMHeadModel.from_pretrained('gpt2')
        self.lm_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.lm.eval()
        
        # Freeze LM (only use for scoring)
        for param in self.lm.parameters():
            param.requires_grad = False
    
    def forward(self, generated_text_batch: List[str]) -> torch.Tensor:
        """
        Compute perplexity of generated text
        Lower perplexity = better grammar
        
        Args:
            generated_text_batch: List of generated report strings
        Returns:
            perplexity_loss: Scalar (lower = better)
        """
        total_perplexity = 0.0
        
        for text in generated_text_batch:
            # Tokenize
            inputs = self.lm_tokenizer(text, return_tensors='pt')
            inputs = {k: v.to(self.lm.device) for k, v in inputs.items()}
            
            # Get LM loss (cross-entropy)
            with torch.no_grad():
                outputs = self.lm(**inputs, labels=inputs['input_ids'])
                lm_loss = outputs.loss
            
            # Perplexity = exp(loss)
            perplexity = torch.exp(lm_loss)
            total_perplexity += perplexity.item()
        
        avg_perplexity = total_perplexity / len(generated_text_batch)
        
        # Normalize to [0, 1] range (lower = better)
        # Good reports have perplexity 10-50
        # Bad reports have perplexity 100-1000
        normalized_loss = min(avg_perplexity / 100.0, 1.0)
        
        return torch.tensor(normalized_loss, device=self.lm.device)
    
# ============================================
# MAIN LOSS: SimplifiedReportGenerationLoss
# (RECOMMENDED - NO FLUENCY)
# ============================================

class SimplifiedReportGenerationLoss(nn.Module):
    def __init__(self, 
                 vocab,
                 lambda_ce: float = 3.0,
                 lambda_uncertainty: float = 0.10,
                 lambda_diversity: float = 0.08,
                 lambda_length: float = 0.05,           # ✅ Keep reduced
                 lambda_repetition: float = 0.15):
        super().__init__()
        
        self.lambda_ce = lambda_ce
        self.lambda_uncertainty = lambda_uncertainty
        self.lambda_diversity = lambda_diversity
        self.lambda_length = lambda_length
        self.lambda_repetition = lambda_repetition
        
        # Loss components
        self.uncertainty_loss = UncertaintyCalibrationLoss()
        self.diversity_loss = DiversityLoss()
        self.repetition_loss = RepetitionPenaltyLoss(ngram_size=3)
        # ❌ REMOVED: self.grammar_loss = GrammarFluencyLoss()
        
        self.latest_losses = {}
        
        print("✅ SimplifiedReportGenerationLoss - ANTI-COLLAPSE VERSION:")
        print(f"  λ_ce: {lambda_ce:.2f}")
        print(f"  λ_uncertainty: {lambda_uncertainty:.2f}")
        print(f"  λ_diversity: {lambda_diversity:.2f}")
        print(f"  λ_length: {lambda_length:.2f}")
        print(f"  λ_repetition: {lambda_repetition:.2f}")
        # ❌ REMOVED lambda_grammar from init
    
    def forward(self, generated_logits, target_tokens, generated_tokens, 
                diagnostic_output, vocab, vocab_size):
        device = generated_logits.device
        
        # 1. Cross-Entropy
        ce_loss = F.cross_entropy(
            generated_logits.reshape(-1, vocab_size),
            target_tokens.reshape(-1),
            ignore_index=vocab.pad_token_id,
            reduction='mean'
        )
        
        # 2. Uncertainty Calibration
        uncertainty_loss = self.uncertainty_loss(
            generated_logits, generated_tokens, diagnostic_output, vocab
        )
        
        # 3. Diversity (returns dict)
        diversity_losses = self.diversity_loss(generated_tokens, generated_logits, vocab_size)
        diversity_loss = diversity_losses['total']
        
        # 4. Length Matching (FIXED)
        gen_lengths = (generated_tokens != vocab.pad_token_id).sum(dim=1).float()
        tgt_lengths = (target_tokens != vocab.pad_token_id).sum(dim=1).float()
        
        # ✅ CRITICAL FIX: Normalize by target length
        length_diff = torch.abs(gen_lengths - tgt_lengths)
        length_loss = (length_diff / (tgt_lengths.mean() + 1e-6)).mean()  # ✅ Add epsilon
        
        # 5. Repetition Penalty
        repetition_loss = self.repetition_loss(generated_tokens)

        # ❌ REMOVED: Grammar loss computation
        
        # Combine (WITHOUT grammar loss)
        total_loss = (
            self.lambda_ce * ce_loss +
            self.lambda_uncertainty * uncertainty_loss +
            self.lambda_diversity * diversity_loss +
            self.lambda_length * length_loss +
            self.lambda_repetition * repetition_loss
            # ❌ REMOVED: + self.lambda_grammar * grammar_loss
        )
        
        # Helper function
        def safe_item(x):
            """Extract float value safely"""
            if isinstance(x, torch.Tensor):
                return x.item()
            return float(x)

        # Store for monitoring
        self.latest_losses = {
            'ce': ce_loss.item(),
            'uncertainty': uncertainty_loss.item(),
            'diversity': diversity_loss.item(),
            'diversity_ngram': safe_item(diversity_losses['ngram_diversity']),
            'diversity_entropy': safe_item(diversity_losses['token_entropy']),
            'diversity_self_bleu': safe_item(diversity_losses['self_bleu']),
            'diversity_overconfidence': safe_item(diversity_losses['overconfidence']),
            'length': length_loss.item(),
            'repetition': repetition_loss.item(),
            'total': total_loss.item()
            # ❌ REMOVED: 'grammar': grammar_loss.item()
        }
        
        return total_loss
    
    def get_latest_losses(self) -> Dict[str, float]:
        """Get component losses for logging"""
        return self.latest_losses.copy()
    
    def get_latest_losses(self) -> Dict[str, float]:
        """Get component losses for logging"""
        return self.latest_losses.copy()


# ============================================
# UTILITY: Label Smoothing (Optional)
# ============================================

def label_smoothing_loss(logits: torch.Tensor, 
                         targets: torch.Tensor,
                         vocab_size: int,
                         smoothing: float = 0.1,
                         ignore_index: int = 0) -> torch.Tensor:
    """
    Label smoothing for better calibration
    
    Args:
        logits: [batch*seq_len, vocab_size]
        targets: [batch*seq_len]
        vocab_size: Size of vocabulary
        smoothing: Smoothing parameter (typically 0.1)
        ignore_index: Padding token ID to ignore
    
    Returns:
        Smoothed cross-entropy loss
    """
    log_probs = F.log_softmax(logits, dim=-1)
    
    # Create smoothed target distribution
    with torch.no_grad():
        confidence = 1.0 - smoothing
        smooth_positives = smoothing / (vocab_size - 1)
        
        # One-hot encoding
        true_dist = torch.zeros_like(log_probs)
        true_dist.fill_(smooth_positives)
        true_dist.scatter_(1, targets.unsqueeze(1), confidence)
        
        # Mask padding
        mask = (targets != ignore_index).float()
    
    # KL divergence
    loss = -(true_dist * log_probs).sum(dim=-1)
    loss = (loss * mask).sum() / mask.sum()
    
    return loss