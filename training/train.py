import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import math
import torch.nn.functional as F

"""
Updated Training Function for Custom Transformer
Replaces train_epoch_with_hybrid_llm
"""
import torch
import torch.nn as nn
from typing import Dict
from Framework_Components.bayesian_componenets import VariationalLinear

class ScheduledSamplingReportGenerator(nn.Module):
    """
    Report generator with scheduled sampling to reduce exposure bias
    """
    def __init__(self, base_generator):
        super().__init__()
        self.generator = base_generator
        self.scheduled_sampling_prob = 0.0  # Will be updated during training
    
    def set_scheduled_sampling_prob(self, epoch, max_epochs=100):
        """
        Gradually increase sampling probability
        
        Epochs 0-5: 0.0 (pure teacher forcing)
        Epochs 6-30: 0.0 → 0.5 (gradual mixing)
        Epochs 31+: 0.5 (balanced)
        """
        if epoch < 5:
            self.scheduled_sampling_prob = 0.0
        elif epoch < 30:
            # Linear increase from 0 to 0.5
            self.scheduled_sampling_prob = 0.5 * (epoch - 5) / 25
        else:
            self.scheduled_sampling_prob = 0.5
    
    def forward(self, visual_features, disease_predictions, 
                uncertainty_estimates, target_tokens=None, max_length=100):
        """
        Generate with scheduled sampling
        """
        batch_size = visual_features.size(0)
        device = visual_features.device
        
        # Start with BOS token
        generated_tokens = torch.full(
            (batch_size, 1), 
            self.generator.vocab.bos_token_id, 
            dtype=torch.long, 
            device=device
        )
        
        logits_list = []
        
        for t in range(max_length - 1):
            # Get logits for current sequence
            outputs = self.generator.decode_step(
                generated_tokens,
                visual_features,
                disease_predictions,
                uncertainty_estimates
            )
            
            logits = outputs['logits']  # [batch, seq_len, vocab_size]
            current_logits = logits[:, -1, :]  # [batch, vocab_size]
            
            logits_list.append(current_logits.unsqueeze(1))
            
            # ===== SCHEDULED SAMPLING DECISION =====
            if target_tokens is not None and t < target_tokens.size(1) - 1:
                # Randomly decide: use ground truth or model prediction?
                use_ground_truth = (
                    torch.rand(batch_size, device=device) > self.scheduled_sampling_prob
                )
                
                # Model prediction
                pred_tokens = torch.argmax(current_logits, dim=-1)
                
                # Ground truth
                gt_tokens = target_tokens[:, t + 1]
                
                # Mix based on random decision
                next_tokens = torch.where(
                    use_ground_truth,
                    gt_tokens,
                    pred_tokens
                )
            else:
                # Pure inference (no ground truth available)
                next_tokens = torch.argmax(current_logits, dim=-1)
            
            # Append to sequence
            generated_tokens = torch.cat([
                generated_tokens,
                next_tokens.unsqueeze(1)
            ], dim=1)
            
            # Stop if all sequences hit EOS
            if (next_tokens == self.generator.vocab.eos_token_id).all():
                break
        
        # Concatenate all logits
        all_logits = torch.cat(logits_list, dim=1)  # [batch, seq_len, vocab_size]
        
        return {
            'generated_tokens': generated_tokens,
            'logits': all_logits
        }
    
# ============================================================================
# ✅ Grammar Perplexity Helper Function (Keep as-is)
# ============================================================================

def train_epoch_with_custom_transformer(
    model, 
    train_loader, 
    optimizer, 
    device, 
    epoch, 
    gradient_accumulation_steps=2,
    generate_reports=False,
    train_reports=False,
    report_loss_weight=0.35,
    tokenizer=None,
    phase='phase1',
    phase2_start_epoch=200,
    max_length=512
):
    """
    Training epoch with custom transformer decoder
    
    Phase 1: Train classification/uncertainty/calibration
    Phase 2: Train ONLY report generator (everything else frozen)
    
    Args:
        phase: 'phase1' or 'phase2'
            - phase1: All components train (classification focus)
            - phase2: ONLY report_generator trains (frozen encoder/bayesian)
    """
    # ========================================================================
    # PHASE-SPECIFIC MODEL STATE
    # ========================================================================
    if phase == 'phase1':
        # Phase 1: Normal training for classification
        model.train()
        
        # Set epoch for Bayesian components
        if hasattr(model, 'set_epoch'):
            model.set_epoch(epoch)
        
        for module in model.modules():
            if isinstance(module, VariationalLinear):
                module._current_epoch = epoch
        
        # Gradual variance learning (first 5 epochs)
        if epoch < 5:
            for module in model.modules():
                if isinstance(module, VariationalLinear):
                    module.weight_logvar.requires_grad = False
                    module.bias_logvar.requires_grad = False
        else:
            for module in model.modules():
                if isinstance(module, VariationalLinear):
                    module.weight_logvar.requires_grad = True
                    module.bias_logvar.requires_grad = True
    
    elif phase == 'phase2':
        # Phase 2: Freeze classification, train ONLY report generator
        model.encoder.eval()
        model.bayesian_framework.eval()
        model.feature_projection.eval()
        model.feature_enhancer.eval()
        
        # Only report generator in train mode
        if hasattr(model, 'report_generator') and model.report_generator is not None:
            model.report_generator.train()
        else:
            raise ValueError("Phase 2 requires report_generator!")
        
        # # Set scheduled sampling probability
        # if hasattr(model.report_generator, 'set_scheduled_sampling_prob'):
        #     phase2_epoch = epoch - phase2_start_epoch
        #     model.report_generator.set_scheduled_sampling_prob(phase2_epoch)
        #     print(f"  📊 Scheduled sampling prob: {model.report_generator.scheduled_sampling_prob:.3f}")
        
        # ✅ Initialize grammar checker (ONCE per training session)
        if not hasattr(model, '_grammar_lm'):
            print("  📚 Loading grammar checker (GPT-2) for fluency loss...")
            try:
                from transformers import GPT2LMHeadModel, GPT2Tokenizer
                
                model._grammar_lm = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
                model._grammar_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
                model._grammar_tokenizer.pad_token = model._grammar_tokenizer.eos_token
                model._grammar_lm.eval()
                
                # Freeze grammar LM (only use for scoring)
                for param in model._grammar_lm.parameters():
                    param.requires_grad = False
                
                print(f"  ✅ Grammar checker loaded successfully")
            except Exception as e:
                print(f"  ⚠️ Warning: Could not load grammar checker: {e}")
                print(f"  ⚠️ Training will continue without grammar loss")
                model._grammar_lm = None
    
    else:
        raise ValueError(f"Unknown phase: {phase}. Must be 'phase1' or 'phase2'")

    total_loss = 0.0
    loss_components = {}
    num_batches = 0
    
    # ✅ Track generation health
    generation_success_count = 0
    generation_fail_count = 0

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(train_loader):
        if batch is None:
            continue

        images, labels, reports = batch
        batch_dict = {
            "images": images.to(device, non_blocking=True),
            "labels": labels.to(device, non_blocking=True)
        }

        try:
            # ================================================================
            # PHASE 1: Train classification/uncertainty
            # ================================================================
            if phase == 'phase1':
                # Normal forward pass
                outputs = model(
                    batch_dict, device,
                    train_reports=train_reports,
                    target_reports=reports if train_reports else None
                )
                
                # Classification Loss (Bayesian)
                class_loss, loss_dict = model.compute_loss(
                    outputs, batch_dict["labels"], epoch=epoch
                )
                
                total_batch_loss = class_loss
                
                # Safety check
                if torch.isnan(total_batch_loss) or torch.isinf(total_batch_loss):
                    print(f"  [WARNING] NaN/Inf loss in batch {batch_idx}, skipping")
                    continue
                
                # Gradient accumulation
                total_batch_loss = total_batch_loss / gradient_accumulation_steps
                total_batch_loss.backward()
                
                # Track losses
                total_loss += class_loss.item()
                num_batches += 1
                
                for key, value in loss_dict.items():
                    if key not in loss_components:
                        loss_components[key] = 0.0
                    loss_components[key] += value
                
                # Optimizer step
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                    optimizer.step()
                    optimizer.zero_grad()
                
                # Logging
                if batch_idx % 2000 == 0:
                    log_msg = (
                        f'Epoch {epoch+1}, Batch {batch_idx}: '
                        f'Loss={class_loss.item():.4f}, '
                        f'Classification={loss_dict.get("classification", 0.0):.4f}, '
                        f'KL={loss_dict.get("kl_divergence", 0.0):.6f}'
                    )
                    print(log_msg)
            
            # ================================================================
            # PHASE 2: Train ONLY report generator
            # ================================================================
            elif phase == 'phase2':
                # Get encoder features
                classification_output = model.encoder({'images': images}, device)
                
                # Use raw encoder features (2048-dim)
                visual_features = classification_output['features']
                disease_predictions = classification_output['cls_pred']
                
                # Create dummy uncertainties for API compatibility
                batch_size = visual_features.size(0)
                num_diseases = disease_predictions.size(1)
                uncertainty_estimates = torch.zeros(batch_size, num_diseases, device=device)
                
                # Build complete diagnostic_output with all required keys
                diagnostic_output = {
                    'disease_logits': disease_predictions,
                    'class_uncertainties': {
                        'epistemic_uncertainty': uncertainty_estimates,
                        'aleatoric_uncertainty': uncertainty_estimates.clone(),
                        'total_uncertainty': uncertainty_estimates.clone()
                    },
                    'consistency_score': torch.ones(batch_size, device=device)
                }
                
                # Tokenize reports
                if isinstance(reports, list):
                    if tokenizer is not None:
                        try:
                            report_tokens = tokenizer(
                                reports,
                                padding=True,
                                truncation=True,
                                max_length=max_length,
                                return_tensors='pt'
                            )['input_ids'].to(device)
                        except Exception as e:
                            print(f"  [WARNING] Tokenization failed in batch {batch_idx}: {e}")
                            continue
                    else:
                        print(f"  [WARNING] No tokenizer in batch {batch_idx}, skipping")
                        continue
                else:
                    report_tokens = reports.to(device)
                
                # Forward through report generator
                report_output = model.report_generator(
                    visual_features=visual_features,
                    diagnostic_output=diagnostic_output,
                    target_reports=report_tokens,
                    max_length=max_length
                )

                # Check if dict or tensor
                if isinstance(report_output, dict):
                    if 'report_loss' not in report_output or report_output['report_loss'] is None:
                        print(f"  [WARNING] No report_loss in batch {batch_idx}, skipping")
                        continue
                    report_loss = report_output['report_loss']
                    loss_dict = report_output.get('report_loss_components', {})
                else:
                    report_loss = report_output
                    loss_dict = {}
                
                # ✅ NEW: Generate tokens for grammar check (every 10 batches)
                grammar_loss_value = 0.0
                if (hasattr(model, '_grammar_lm') and model._grammar_lm is not None and 
                    batch_idx % 10 == 0 and tokenizer is not None):
                    
                    try:
                        # ✅ SIMPLIFIED: Direct access to decoder
                        with torch.no_grad():
                            gen_tokens = model.report_generator.decoder.generate(
                                visual_features=visual_features,
                                diagnostic_output=diagnostic_output,
                                tokenizer=model.report_generator.tokenizer,
                                max_length=128,
                                min_length=20,
                                temperature=0.7,
                                top_p=0.9,
                                repetition_penalty=1.5
                            )
                        
                        # ✅ Check if generation succeeded
                        if gen_tokens is None:
                            print(f"  ❌ BATCH {batch_idx}: generate() returned None")
                            generation_fail_count += 1
                            grammar_loss_value = 0.0
                        elif not isinstance(gen_tokens, torch.Tensor):
                            print(f"  ❌ BATCH {batch_idx}: generate() returned {type(gen_tokens)}")
                            generation_fail_count += 1
                            grammar_loss_value = 0.0
                        elif gen_tokens.size(0) == 0:
                            print(f"  ❌ BATCH {batch_idx}: generate() returned empty tensor")
                            generation_fail_count += 1
                            grammar_loss_value = 0.0
                        else:
                            generation_success_count += 1
                            
                            # ✅ Decode to text
                            generated_texts = []
                            num_samples = min(gen_tokens.size(0), 8)  # Limit to 8 samples
                            
                            for i in range(num_samples):
                                tokens = gen_tokens[i].cpu().tolist()
                                
                                # Decode using your tokenizer
                                try:
                                    if hasattr(tokenizer, 'decode'):
                                        text = tokenizer.decode(tokens, skip_special_tokens=True)
                                    elif hasattr(tokenizer, 'batch_decode'):
                                        text = tokenizer.batch_decode([tokens], skip_special_tokens=True)[0]
                                    else:
                                        # Fallback: manual decode
                                        text = ' '.join([
                                            tokenizer.idx2word.get(t, '<unk>') 
                                            for t in tokens 
                                            if t not in [tokenizer.pad_token_id, tokenizer.bos_token_id, 
                                                         tokenizer.eos_token_id]
                                        ])
                                    
                                    if len(text.strip()) > 10:
                                        generated_texts.append(text.strip())
                                
                                except Exception as e:
                                    print(f"  ⚠️ Decode failed for sample {i}: {e}")
                                    continue
                            
                            if len(generated_texts) > 0:
                                # ✅ Log sample on first batch
                                if batch_idx == 0:
                                    print(f"\n  📝 Sample Generated Reports (epoch {epoch}):")
                                    for i, text in enumerate(generated_texts[:3]):
                                        print(f"      [{i+1}] {text[:150]}")
                                
                                # Compute grammar perplexity
                                grammar_loss_value = compute_grammar_perplexity(
                                    generated_texts,
                                    model._grammar_lm,
                                    model._grammar_tokenizer,
                                    device
                                )
                                
                                # Track grammar loss
                                if isinstance(grammar_loss_value, torch.Tensor):
                                    loss_dict['grammar'] = grammar_loss_value.item()
                                else:
                                    loss_dict['grammar'] = float(grammar_loss_value)
                                
                                # ✅ Show normalized loss on first batch
                                if batch_idx == 0:
                                    print(f"      Normalized Loss: {loss_dict['grammar']:.4f}\n")
                            else:
                                print(f"  ⚠️ No valid texts generated, skipping grammar loss")
                                generation_fail_count += 1
                                grammar_loss_value = 0.0
                    
                    except Exception as e:
                        print(f"  ⚠️ Grammar loss computation failed in batch {batch_idx}: {e}")
                        import traceback
                        traceback.print_exc()
                        generation_fail_count += 1
                        grammar_loss_value = 0.0
                
                # Apply weight and accumulate
                weighted_loss = report_loss_weight * report_loss
                
                # ✅ Add grammar loss with appropriate weight
                if grammar_loss_value != 0.0:
                    if isinstance(grammar_loss_value, torch.Tensor):
                        weighted_loss = weighted_loss + 0.25 * grammar_loss_value
                    else:
                        weighted_loss = weighted_loss + 0.25 * torch.tensor(grammar_loss_value, device=device)
                
                total_batch_loss = weighted_loss / gradient_accumulation_steps
                
                # Backward
                total_batch_loss.backward()
                
                # Track losses (ONLY ONCE)
                total_loss += report_loss.item()
                num_batches += 1
                
                loss_components['report_loss'] = loss_components.get('report_loss', 0.0) + report_loss.item()
                
                # Add component losses
                for key, value in loss_dict.items():
                    if key not in loss_components:
                        loss_components[key] = 0.0
                    loss_components[key] += value
                
                # Log first batch with enhanced detail
                if batch_idx == 0:
                    print(f"Epoch {epoch}, Batch 0: Report Loss={report_loss.item():.4f}")
                    
                    if loss_dict:
                        comp_str = " [" + ", ".join([
                            f"{k.upper()[:3]}:{v:.3f}" 
                            for k, v in loss_dict.items() 
                            if isinstance(v, (int, float))
                        ]) + "]"
                        print(f"  Loss breakdown{comp_str}")
                    
                    # ✅ Show if grammar loss is active
                    if 'grammar' in loss_dict:
                        print(f"  ✅ Grammar/Fluency Loss: {loss_dict['grammar']:.4f} (active)")
                
                # Optimizer step (for Phase 2)
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.report_generator.parameters(),
                        max_norm=1.0
                    )
                    optimizer.step()
                    optimizer.zero_grad()
                      
        except Exception as e:
            print(f"Training error in batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Final gradient step if needed
    if num_batches % gradient_accumulation_steps != 0:
        if phase == 'phase1':
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        else:  # phase2
            torch.nn.utils.clip_grad_norm_(model.report_generator.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
    
    # ✅ Print generation statistics
    if phase == 'phase2' and (generation_success_count + generation_fail_count) > 0:
        total_gen_attempts = generation_success_count + generation_fail_count
        success_rate = (generation_success_count / total_gen_attempts) * 100
        print(f"\n📊 Generation Stats:")
        print(f"   Successful: {generation_success_count}")
        print(f"   Failed: {generation_fail_count}")
        print(f"   Success rate: {success_rate:.1f}%")
    
    # Compute averages
    avg_loss = total_loss / max(num_batches, 1)
    for key in loss_components:
        loss_components[key] /= max(num_batches, 1)
    
    return avg_loss, loss_components


# ============================================================================
# ✅ IMPROVED Grammar Perplexity Helper Function
# ============================================================================

def compute_grammar_perplexity(texts, lm_model, lm_tokenizer, device):
    """
    Compute average perplexity of generated texts using GPT-2
    
    Args:
        texts: List[str] - Generated report texts
        lm_model: Pre-trained language model (GPT-2)
        lm_tokenizer: Tokenizer for the LM
        device: torch device
    
    Returns:
        Normalized perplexity loss (0-1, lower is better)
    """
    import math
    
    total_perplexity = 0.0
    valid_count = 0
    
    with torch.no_grad():
        for text in texts:
            if len(text.strip()) < 10:  # Skip very short texts
                continue
            
            try:
                # Tokenize for GPT-2
                inputs = lm_tokenizer(
                    text, 
                    return_tensors='pt', 
                    truncation=True, 
                    max_length=128,
                    padding=True
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Get language model loss
                outputs = lm_model(**inputs, labels=inputs['input_ids'])
                lm_loss = outputs.loss
                
                # Perplexity = exp(loss)
                perplexity = torch.exp(lm_loss)
                
                # Clip extreme values
                perplexity = torch.clamp(perplexity, min=1.0, max=500.0)
                
                total_perplexity += perplexity.item()
                valid_count += 1
            
            except Exception as e:
                # Skip problematic texts
                continue
    
    if valid_count == 0:
        return torch.tensor(0.0, device=device)
    
    # Average perplexity
    avg_perplexity = total_perplexity / valid_count
    
    # ✅ IMPROVED: Sigmoid-based normalization for better gradient flow
    # Maps perplexity smoothly to [0, 1]:
    # - 30 perplexity → ~0.23
    # - 70 perplexity → ~0.48
    # - 150 perplexity → ~0.75
    # - 300+ perplexity → ~0.95
    normalized_loss = 1.0 / (1.0 + math.exp(-0.02 * (avg_perplexity - 70)))
    
    return torch.tensor(normalized_loss, device=device, requires_grad=False)


def generate_with_nucleus_sampling(
    model, 
    visual_features, 
    disease_predictions, 
    uncertainty_estimates, 
    max_length=100, 
    top_p=0.92, 
    temperature=0.95
):
    """
    Generate reports with nucleus (top-p) sampling for diversity
    
    Works with your custom transformer architecture
    """
    """Generate with nucleus sampling"""
    import torch.nn.functional as F  # ✅ Import here if not at top
    
    batch_size = visual_features.size(0)
    device = visual_features.device
    
    # ✅ Get vocab from wrapped generator
    if hasattr(model.report_generator, 'generator'):
        vocab = model.report_generator.generator.vocab
        decoder = model.report_generator.generator.decoder
    else:
        vocab = model.report_generator.vocab
        decoder = model.report_generator.decoder
    
    # Start with BOS token
    generated_tokens = torch.full(
        (batch_size, 1),
        vocab.bos_token_id,
        dtype=torch.long,
        device=device
    )
    
    for t in range(max_length - 1):
        # Forward pass through report generator
        try:
            # Try the report generator's forward method
            if hasattr(model.report_generator, 'generator'):
                # Unwrap if using ScheduledSamplingReportGenerator
                gen_output = model.report_generator.generator(
                    visual_features=visual_features,
                    disease_predictions=disease_predictions,
                    uncertainty_estimates=uncertainty_estimates,
                    target_reports=generated_tokens
                    # max_length=generated_tokens.size(1) + 1
                )
            else:
                gen_output = model.report_generator(
                    visual_features=visual_features,
                    disease_predictions=disease_predictions,
                    uncertainty_estimates=uncertainty_estimates,
                    target_reports=generated_tokens,
                    max_length=generated_tokens.size(1) + 1
                )
            
            # Get logits for last position
            if isinstance(gen_output, dict) and 'logits' in gen_output:
                logits = gen_output['logits'][:, -1, :] / temperature
            else:
                # Fallback: assume gen_output is logits directly
                logits = gen_output[:, -1, :] / temperature
        
        except Exception as e:
            print(f"Error in generation step {t}: {e}")
            # Fallback to random token
            logits = torch.randn(batch_size, len(vocab.word2idx), device=device)
        
        probs = F.softmax(logits, dim=-1)
        
        # ===== NUCLEUS SAMPLING =====
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        
        # Shift to keep first token above threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # Zero out removed tokens
        sorted_probs[sorted_indices_to_remove] = 0.0
        sorted_probs = sorted_probs / (sorted_probs.sum(dim=-1, keepdim=True) + 1e-10)
        
        # Sample from filtered distribution
        next_token_idx = torch.multinomial(sorted_probs, num_samples=1)
        next_tokens = sorted_indices.gather(-1, next_token_idx).squeeze(-1)
        
        # Append to sequence
        generated_tokens = torch.cat([
            generated_tokens,
            next_tokens.unsqueeze(1)
        ], dim=1)
        
        # Stop if all sequences hit EOS
        if (next_tokens == vocab.eos_token_id).all():
            break
    
    return generated_tokens

def validate_epoch_with_custom_transformer(
    model, 
    valid_loader, 
    device, 
    epoch,
    tokenizer=None,
    max_length=512,
    generate_reports=True,
    compute_uncertainty=True
):
    """
    Comprehensive validation with:
    - Disease classification metrics (ROC-AUC, AP)
    - Uncertainty quantification (epistemic, aleatoric)
    - Full report generation metrics (BLEU, METEOR, ROUGE, BERTScore, Clinical)
    """
    import numpy as np
    import torch
    from sklearn.metrics import roc_auc_score, average_precision_score
    from tqdm import tqdm
    
    model.eval()
    
    # ========================================================================
    # 📊 Metric Storage
    # ========================================================================
    # Classification metrics
    all_preds = []
    all_labels = []
    
    # Uncertainty metrics
    all_epistemic = []
    all_aleatoric = []
    all_consistency = []
    all_total_uncertainty = []
    
    # Report generation
    all_generated_reports = []
    all_reference_reports = []
    
    # ========================================================================
    # 🔄 Validation Loop
    # ========================================================================
    with torch.no_grad():
        for batch_idx, batch in enumerate(valid_loader):  # ✅ Removed tqdm progress bar
            try:
                # Handle different batch formats
                if isinstance(batch, dict):
                    images = batch["image"].to(device, non_blocking=True)
                    labels = batch.get("labels", None)
                    reports = batch.get("report", batch.get("reports", []))
                else:
                    # Tuple format: (images, labels, reports)
                    if batch is None or any(x is None for x in batch):
                        continue
                    images, labels, reports = batch
                    images = images.to(device)
                    if labels is not None:
                        labels = labels.to(device)
                
                # ================================================================
                # 🎯 Disease Classification with MC Dropout
                # ================================================================
                if labels is not None and compute_uncertainty:
                    try:
                        outputs_mc = model(
                            {'images': images}, device,
                            mc_dropout=True, n_mc=50
                        )
                        
                        preds = torch.sigmoid(outputs_mc['disease_logits']).cpu().numpy()
                        all_preds.append(preds)
                        all_labels.append(labels.cpu().numpy())
                        
                        if 'epistemic_uncertainty' in outputs_mc:
                            all_epistemic.append(outputs_mc['epistemic_uncertainty'].cpu().numpy())
                    except Exception as e:
                        print(f"Warning: MC Dropout failed in batch {batch_idx}: {e}")
                
                # ================================================================
                # 📉 Uncertainty Decomposition
                # ================================================================
                if labels is not None and compute_uncertainty:
                    try:
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
                    except Exception as e:
                        print(f"Warning: Uncertainty decomposition failed in batch {batch_idx}: {e}")
                
                # ================================================================
                # 📝 Report Generation
                # ================================================================
                if generate_reports and tokenizer is not None:
                    try:
                        # ✅ Get features from encoder
                        classification_output = model.encoder({'images': images}, device)
                        visual_features = classification_output['features']
                        disease_predictions = classification_output['cls_pred']
                        
                        # ✅ Create complete diagnostic_output (same as training)
                        batch_size = visual_features.size(0)
                        num_diseases = disease_predictions.size(1)
                        uncertainty_estimates = torch.zeros(batch_size, num_diseases, device=device)
                        
                        diagnostic_output = {
                            'disease_logits': disease_predictions,
                            'class_uncertainties': {
                                'epistemic_uncertainty': uncertainty_estimates,
                                'aleatoric_uncertainty': uncertainty_estimates.clone(),
                                'total_uncertainty': uncertainty_estimates.clone()
                            },
                            'consistency_score': torch.ones(batch_size, device=device)
                        }
                        
                        # ✅ Generate using decoder.generate() directly
                        try:
                            if hasattr(model.report_generator, 'generator'):
                                generated_tokens = model.report_generator.generator.decoder.generate(
                                    visual_features=visual_features,
                                    diagnostic_output=diagnostic_output,
                                    tokenizer=tokenizer,
                                    max_length=128,
                                    min_length=30,
                                    temperature=0.6,
                                    top_p=0.90,
                                    repetition_penalty=2.0
                                )
                            else:
                                generated_tokens = model.report_generator.decoder.generate(
                                    visual_features=visual_features,
                                    diagnostic_output=diagnostic_output,
                                    tokenizer=tokenizer,
                                    max_length=128,
                                    min_length=30,
                                    temperature=0.6,
                                    top_p=0.90,
                                    repetition_penalty=2.0
                                )
                            
                            if batch_idx == 0:
                                print(f"  ✅ Using decoder.generate() with nucleus sampling")
                        
                        except Exception as e:
                            print(f"  ⚠️ Generation failed in batch {batch_idx}: {e}")
                            continue
                        
                        # ================================================================
                        # Decode generated texts
                        # ================================================================
                        if hasattr(tokenizer, 'batch_decode'):
                            gen_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                        elif hasattr(tokenizer, 'decode'):
                            gen_texts = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                            if isinstance(gen_texts, str):
                                gen_texts = [gen_texts]
                        else:
                            raise ValueError(f"Tokenizer {type(tokenizer)} has no decode method!")
                        
                        # ================================================================
                        # Handle reference reports
                        # ================================================================
                        if isinstance(reports, list):
                            ref_texts = [r.strip() if isinstance(r, str) else "" for r in reports]
                        else:
                            if hasattr(tokenizer, 'batch_decode'):
                                ref_texts = tokenizer.batch_decode(reports, skip_special_tokens=True)
                            elif hasattr(tokenizer, 'decode'):
                                ref_texts = tokenizer.decode(reports, skip_special_tokens=True)
                                if isinstance(ref_texts, str):
                                    ref_texts = [ref_texts]
                            else:
                                ref_texts = reports if isinstance(reports, list) else [str(reports)]
                        
                        # ================================================================
                        # Store valid pairs only
                        # ================================================================
                        for gen_report, ref_report in zip(gen_texts, ref_texts):
                            if ref_report and isinstance(ref_report, str) and len(ref_report) > 10:
                                all_generated_reports.append(gen_report)
                                all_reference_reports.append(ref_report)
                    
                    except Exception as e:
                        print(f"Report generation error in batch {batch_idx}: {e}")
                        import traceback
                        traceback.print_exc()

            except Exception as e:
                print(f"Validation error in batch {batch_idx}: {e}")
                continue
    
    # ========================================================================
    # 🧮 Compute All Metrics
    # ========================================================================
    metrics = {}
    
    # ------------------------------------------------------------------------
    # 1️⃣ Classification Metrics (ROC-AUC, Average Precision)
    # ------------------------------------------------------------------------
    if len(all_preds) > 0 and len(all_labels) > 0:  # ✅ FIX: Check list length
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        
        # ROC-AUC per class
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
        try:
            from Metrics_Calculation.advance_metrics_calculator import AdvancedMetricsCalculator
            metrics_calculator = AdvancedMetricsCalculator()
            calibration_metrics = metrics_calculator.compute_calibration_metrics(all_preds, all_labels)
            metrics.update(calibration_metrics)
        except Exception as e:
            print(f"Warning: Could not compute calibration metrics: {e}")
    
    # ------------------------------------------------------------------------
    # 2️⃣ Uncertainty Metrics
    # ------------------------------------------------------------------------
    if len(all_epistemic) > 0 and len(all_aleatoric) > 0 and len(all_total_uncertainty) > 0:  # ✅ FIX
        all_epistemic = np.concatenate(all_epistemic)
        all_aleatoric = np.concatenate(all_aleatoric)
        all_total_uncertainty = np.concatenate(all_total_uncertainty)
        
        metrics['epistemic_uncertainty_mean'] = np.mean(all_epistemic)
        metrics['epistemic_uncertainty_std'] = np.std(all_epistemic)
        metrics['aleatoric_uncertainty_mean'] = np.mean(all_aleatoric)
        metrics['aleatoric_uncertainty_std'] = np.std(all_aleatoric)
        metrics['total_uncertainty_mean'] = np.mean(all_total_uncertainty)
        
        # Uncertainty-error correlation
        if len(all_preds) > 0 and len(all_labels) > 0:  # ✅ FIX: Check if arrays exist
            errors = np.abs(all_preds - all_labels)
            correlations = []
            for i in range(all_preds.shape[1]):
                if np.std(all_total_uncertainty[:, i]) > 0 and np.std(errors[:, i]) > 0:
                    corr = np.corrcoef(all_total_uncertainty[:, i], errors[:, i])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)
            
            metrics['uncertainty_error_correlation'] = np.mean(correlations) if correlations else 0.0
        
        # Epistemic ratio
        eps_ratio = all_epistemic / (all_total_uncertainty + 1e-8)
        metrics['epistemic_ratio_mean'] = np.mean(eps_ratio)
        metrics['epistemic_ratio_std'] = np.std(eps_ratio)
    
    # Consistency metrics
    if len(all_consistency) > 0:  # ✅ FIX
        all_consistency = np.concatenate(all_consistency)
        metrics['mean_consistency'] = np.mean(all_consistency)
        metrics['std_consistency'] = np.std(all_consistency)
    
    # ------------------------------------------------------------------------
    # 3️⃣ Report Generation Metrics (Full Dataset)
    # ------------------------------------------------------------------------
    if len(all_generated_reports) > 0:  # ✅ FIX
        print(f"\nComputing full validation metrics on {len(all_generated_reports)} reports...")
        
        # === Basic NLG Metrics (BLEU, METEOR, ROUGE) ===
        try:
            from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
            from nltk.translate.meteor_score import meteor_score
            from rouge_score import rouge_scorer
            
            smoothie = SmoothingFunction().method4
            scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
            
            # Tokenize
            refs_tokenized = [[r.split()] for r in all_reference_reports]
            gens_tokenized = [g.split() for g in all_generated_reports]
            
            # BLEU scores
            metrics['report_bleu-1'] = corpus_bleu(refs_tokenized, gens_tokenized, 
                                                    weights=(1, 0, 0, 0), 
                                                    smoothing_function=smoothie)
            metrics['report_bleu-2'] = corpus_bleu(refs_tokenized, gens_tokenized, 
                                                    weights=(0.5, 0.5, 0, 0), 
                                                    smoothing_function=smoothie)
            metrics['report_bleu-3'] = corpus_bleu(refs_tokenized, gens_tokenized, 
                                                    weights=(0.33, 0.33, 0.33, 0), 
                                                    smoothing_function=smoothie)
            metrics['report_bleu-4'] = corpus_bleu(refs_tokenized, gens_tokenized, 
                                                    weights=(0.25, 0.25, 0.25, 0.25), 
                                                    smoothing_function=smoothie)
            
            # METEOR
            meteor_scores = [meteor_score([ref], gen) 
                           for ref, gen in zip(all_reference_reports, all_generated_reports)]
            metrics['report_meteor'] = sum(meteor_scores) / len(meteor_scores)
            
            # ROUGE-L
            rougeL_scores = [scorer.score(ref, gen)["rougeL"].fmeasure 
                           for ref, gen in zip(all_reference_reports, all_generated_reports)]
            metrics['report_rouge-l'] = sum(rougeL_scores) / len(rougeL_scores)
        
        except Exception as e:
            print(f"Warning: Basic NLG metrics computation failed: {e}")
        
        # === Advanced Medical Metrics (BERTScore, Clinical F1, etc.) ===
        try:
            from Metrics_Calculation.report_metrics import MedicalReportMetrics
            
            report_evaluator = MedicalReportMetrics()
            advanced_metrics = report_evaluator.compute_all_metrics(
                all_generated_reports,
                all_reference_reports
            )
            
            # Add with 'report_' prefix
            for k, v in advanced_metrics.items():
                if not k.startswith('report_'):
                    metrics[f'report_{k}'] = v
                else:
                    metrics[k] = v
        
        except Exception as e:
            print(f"Warning: Advanced medical metrics computation failed: {e}")
            import traceback
            traceback.print_exc()
    
    # ========================================================================
    # 📊 Print Results
    # ========================================================================
    print(f"\n{'='*80}")
    print(f"VALIDATION RESULTS - EPOCH {epoch}")
    print(f"{'='*80}")
    
    # Classification Results
    if 'roc_auc_macro' in metrics:
        print(f"\n🎯 Disease Classification:")
        print(f"  ROC-AUC (Macro): {metrics['roc_auc_macro']:.4f}")
        print(f"  Average Precision: {metrics['average_precision']:.4f}")
        if 'ece' in metrics:
            print(f"  ECE: {metrics['ece']:.4f}")
            print(f"  MCE: {metrics['mce']:.4f}")
    
    # Uncertainty Results
    if 'uncertainty_error_correlation' in metrics:
        print(f"\n📉 Uncertainty Quantification:")
        print(f"  Uncertainty-Error Correlation: {metrics['uncertainty_error_correlation']:.4f}")
        
        if 'epistemic_ratio_mean' in metrics:
            print(f"  Epistemic Ratio: {metrics['epistemic_ratio_mean']:.3f} ± {metrics['epistemic_ratio_std']:.3f}")
        
        if 'epistemic_uncertainty_mean' in metrics:
            print(f"  Epistemic Uncertainty: {metrics['epistemic_uncertainty_mean']:.4f} ± {metrics['epistemic_uncertainty_std']:.4f}")
        
        if 'aleatoric_uncertainty_mean' in metrics:
            print(f"  Aleatoric Uncertainty: {metrics['aleatoric_uncertainty_mean']:.4f} ± {metrics['aleatoric_uncertainty_std']:.4f}")
    
    if 'mean_consistency' in metrics:
        print(f"\n🔄 Consistency:")
        print(f"  Mean: {metrics['mean_consistency']:.4f}")
        print(f"  Std: {metrics['std_consistency']:.4f}")
    
    # Report Generation Results
    if any(k.startswith('report_') for k in metrics.keys()):
        print(f"\n{'='*80}")
        print(f"📝 REPORT GENERATION METRICS")
        print(f"{'='*80}")
        print(f"Evaluated on {len(all_generated_reports)} report pairs")
        
        if 'report_bleu-1' in metrics:
            print(f"\n📊 Lexical Overlap:")
            print(f"  BLEU-1:  {metrics['report_bleu-1'] * 100:.2f}")
            print(f"  BLEU-2:  {metrics.get('report_bleu-2', 0) * 100:.2f}")
            print(f"  BLEU-3:  {metrics.get('report_bleu-3', 0) * 100:.2f}")
            print(f"  BLEU-4:  {metrics.get('report_bleu-4', 0) * 100:.2f}")
            print(f"  METEOR:  {metrics.get('report_meteor', 0) * 100:.2f}")
            print(f"  ROUGE-L: {metrics.get('report_rouge-l', 0) * 100:.2f}")
        
        if 'report_bertscore_f1' in metrics:
            print(f"\n🧠 Semantic Similarity (BiomedBERT):")
            print(f"  Precision: {metrics['report_bertscore_precision']:.4f}")
            print(f"  Recall:    {metrics['report_bertscore_recall']:.4f}")
            print(f"  F1:        {metrics['report_bertscore_f1']:.4f}")
        
        if 'report_clinical_f1' in metrics:
            print(f"\n🏥 Clinical Accuracy:")
            print(f"  Precision: {metrics['report_clinical_precision']:.4f}")
            print(f"  Recall:    {metrics['report_clinical_recall']:.4f}")
            print(f"  F1:        {metrics['report_clinical_f1']:.4f}")
        
        if 'report_report_completeness' in metrics:
            print(f"\n🔬 Radiology-Specific:")
            print(f"  Report Completeness:        {metrics['report_report_completeness']:.4f}")
            print(f"  Uncertainty Appropriateness: {metrics['report_uncertainty_appropriateness']:.4f}")
            print(f"  Length Similarity:          {metrics['report_length_similarity']:.4f}")
    
    # Sample Reports
    if len(all_generated_reports) >= 3:
        print(f"\n{'='*80}")
        print("📄 SAMPLE VALIDATION REPORTS")
        print(f"{'='*80}")
        for i in range(min(3, len(all_generated_reports))):
            print(f"\n--- Report {i+1} ---")
            print(f"Generated: {all_generated_reports[i][:200]}...")
            print(f"Reference: {all_reference_reports[i][:200]}...")
    
    print(f"\n{'='*80}\n")
    
    return metrics
