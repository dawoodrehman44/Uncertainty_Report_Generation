"""
Custom Transformer Decoder with Explicit Uncertainty Encoding
Trains from scratch on MIMIC-CXR reports
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict
from typing import Union, Dict

class UncertaintyEncoder(nn.Module):
    """Encode Bayesian uncertainties into the generation process"""
    
    def __init__(self, hidden_dim: int = 512):
        super().__init__()
        
        # Separate encoders for different uncertainty types
        self.epistemic_encoder = nn.Sequential(
            nn.Linear(14, hidden_dim // 4),  # 14 diseases
            nn.LayerNorm(hidden_dim // 4),
            nn.GELU()
        )
        
        self.aleatoric_encoder = nn.Sequential(
            nn.Linear(14, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.GELU()
        )
        
        # Consistency score encoder
        self.consistency_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.GELU()
        )
        
        # Disease logits encoder
        self.logits_encoder = nn.Sequential(
            nn.Linear(14, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.GELU()
        )
        
        # Combine all uncertainty information
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
    
    def forward(self, diagnostic_output: Dict) -> torch.Tensor:
        """
        Args:
            diagnostic_output: Dict with keys:
                - disease_logits: [batch, 14]
                - class_uncertainties: Dict with epistemic/aleatoric [batch, 14]
                - consistency_score: [batch] or [batch, 1]
        
        Returns:
            uncertainty_embedding: [batch, hidden_dim]
        """
        batch_size = diagnostic_output['disease_logits'].size(0)
        
        # Extract components
        logits = diagnostic_output['disease_logits']
        epistemic = diagnostic_output['class_uncertainties']['epistemic_uncertainty']
        aleatoric = diagnostic_output['class_uncertainties']['aleatoric_uncertainty']
        consistency = diagnostic_output['consistency_score']
        
        # ===== FIX: Handle consistency_score shape =====
        # consistency_score can be [batch] or [batch, 1]
        if consistency.dim() == 1:
            consistency = consistency.unsqueeze(-1)  # [batch] → [batch, 1]
        elif consistency.dim() == 2 and consistency.size(1) != 1:
            # If [batch, N] where N > 1, take mean
            consistency = consistency.mean(dim=-1, keepdim=True)
        # Now consistency is guaranteed to be [batch, 1]
        
        # Encode each component
        logits_emb = self.logits_encoder(logits)          # [batch, d_model/4]
        epistemic_emb = self.epistemic_encoder(epistemic)  # [batch, d_model/4]
        aleatoric_emb = self.aleatoric_encoder(aleatoric)  # [batch, d_model/4]
        consistency_emb = self.consistency_encoder(consistency)  # [batch, d_model/4]
        
        # Concatenate - all are now [batch, d_model/4]
        combined = torch.cat([
            logits_emb, epistemic_emb, aleatoric_emb, consistency_emb
        ], dim=-1)  # [batch, d_model]
        
        # Fuse
        uncertainty_embedding = self.fusion(combined)  # [batch, hidden_dim]
        
        return uncertainty_embedding


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):  # ✅ Increased from 64/128 to 512
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)  # [max_len, 1, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape [seq_len, batch_size, d_model]
        """
        # ✅ CRITICAL FIX: Handle sequences longer than max_len
        seq_len = x.size(0)
        if seq_len > self.pe.size(0):
            # Dynamically expand positional encoding if needed
            raise RuntimeError(
                f"Sequence length {seq_len} exceeds max positional encoding length {self.pe.size(0)}. "
                f"Increase max_len in PositionalEncoding initialization."
            )
        
        x = x + self.pe[:seq_len]  # ✅ Use only the needed part
        return self.dropout(x)



class UncertaintyAwareTransformerDecoder(nn.Module):
    """
    Custom Transformer Decoder with REAL ATTENTION EXTRACTION
    """
    
    def __init__(self, 
                 vocab_size: int,
                 visual_dim: int = 2048,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 max_seq_len: int = 256):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)
        
        # Visual feature adapter
        self.visual_adapter = nn.Sequential(
            nn.Linear(visual_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Uncertainty encoder
        self.uncertainty_encoder = UncertaintyEncoder(d_model)
        
        # Combine visual + uncertainty
        self.memory_fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        
        # Transformer decoder layers
        self.layers = nn.ModuleList([
            TransformerDecoderBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
        # ✅ NEW: Store last attention weights
        self.last_attention_weights = None
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize parameters"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate causal mask for autoregressive generation"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(self, 
                tgt_tokens: torch.Tensor,
                visual_features: torch.Tensor,
                diagnostic_output: Dict,
                tgt_key_padding_mask: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> Union[torch.Tensor, Dict]:
        """
        Training forward pass with OPTIONAL ATTENTION RETURN
        
        Args:
            tgt_tokens: [batch, seq_len] - target token IDs
            visual_features: [batch, visual_dim] - from encoder
            diagnostic_output: Dict with uncertainties
            tgt_key_padding_mask: [batch, seq_len] - padding mask
            return_attention: ✅ NEW - Whether to return attention weights
        
        Returns:
            logits: [batch, seq_len, vocab_size] OR
            dict with {'logits': ..., 'attention_weights': ...}
        """
        batch_size, seq_len = tgt_tokens.size()
        
        # Embed tokens
        tgt_emb = self.token_embedding(tgt_tokens) * math.sqrt(self.d_model)
        tgt_emb = tgt_emb.permute(1, 0, 2)  # [seq_len, batch, d_model]
        tgt_emb = self.pos_encoder(tgt_emb)
        tgt_emb = self.dropout(tgt_emb)
        
        # Prepare memory (visual + uncertainty)
        # ✅ Handle spatial visual features
        if visual_features.dim() == 4:
            # [B, C, H, W] - reshape to [B, H*W, C]
            B, C, H, W = visual_features.shape
            visual_features = visual_features.view(B, C, H*W).permute(0, 2, 1)  # [B, 49, C]
        elif visual_features.dim() == 2:
            # [B, C] - add spatial dimension
            visual_features = visual_features.unsqueeze(1)  # [B, 1, C]

        # Now visual_features is [B, spatial_dim, C]
        visual_emb = self.visual_adapter(visual_features)  # [B, spatial_dim, d_model]

        # Uncertainty (keep as single vector)
        uncertainty_emb = self.uncertainty_encoder(diagnostic_output)  # [B, d_model]
        uncertainty_emb = uncertainty_emb.unsqueeze(1)  # [B, 1, d_model]

        # Concatenate uncertainty to all spatial locations
        B, S, D = visual_emb.shape
        uncertainty_expanded = uncertainty_emb.expand(B, S, D)  # [B, spatial_dim, d_model]

        # Fuse visual and uncertainty
        memory = torch.cat([visual_emb, uncertainty_expanded], dim=-1)  # [B, spatial_dim, 2*d_model]
        memory = self.memory_fusion(memory)  # [B, spatial_dim, d_model]
        memory = memory.permute(1, 0, 2)  # [spatial_dim, B, d_model] - transformer format
        
        # Generate causal mask
        tgt_mask = self._generate_square_subsequent_mask(seq_len).to(tgt_tokens.device)
        
        # ✅ MODIFIED: Pass through decoder layers with attention capture
        output = tgt_emb
        cross_attention_weights = None
        
        for i, layer in enumerate(self.layers):
            # Get attention from LAST layer only
            if return_attention and i == len(self.layers) - 1:
                # Call layer with return_attention=True
                layer_output = layer(output, memory, tgt_mask, tgt_key_padding_mask, return_attention=True)
                
                # Handle different return types
                if isinstance(layer_output, tuple):
                    output, cross_attn = layer_output
                    cross_attention_weights = cross_attn
                else:
                    output = layer_output
            else:
                output = layer(output, memory, tgt_mask, tgt_key_padding_mask)
        
        # Store attention weights
        self.last_attention_weights = cross_attention_weights
        
        # Project to vocabulary
        output = output.permute(1, 0, 2)  # [batch, seq_len, d_model]
        logits = self.output_proj(output)  # [batch, seq_len, vocab_size]
        
        # ✅ Return dict if attention requested
        if return_attention and cross_attention_weights is not None:
            return {
                'logits': logits,
                'attention_weights': cross_attention_weights  # [batch, nhead, seq_len, mem_len]
            }
        
        return logits
    
    @torch.no_grad()
    def generate(self,
                visual_features: torch.Tensor,
                diagnostic_output: Dict,
                tokenizer,
                max_length: int = 60,
                min_length: int = 30,
                temperature: float = 0.9,
                top_p: float = 0.92,
                repetition_penalty: float = 1.3,
                length_penalty: float = 1.2):
        """
        Autoregressive generation with ATTENTION TRACKING
        """
        batch_size = visual_features.size(0)
        device = visual_features.device
        
        # Prepare memory (visual + uncertainty)
        # ✅ Handle spatial visual features
        if visual_features.dim() == 4:
            # [B, C, H, W] - reshape to [B, H*W, C]
            B, C, H, W = visual_features.shape
            visual_features = visual_features.view(B, C, H*W).permute(0, 2, 1)  # [B, 49, C]
        elif visual_features.dim() == 2:
            # [B, C] - add spatial dimension
            visual_features = visual_features.unsqueeze(1)  # [B, 1, C]

        # Now visual_features is [B, spatial_dim, C]
        visual_emb = self.visual_adapter(visual_features)  # [B, spatial_dim, d_model]

        # Uncertainty (keep as single vector)
        uncertainty_emb = self.uncertainty_encoder(diagnostic_output)  # [B, d_model]
        uncertainty_emb = uncertainty_emb.unsqueeze(1)  # [B, 1, d_model]

        # Concatenate uncertainty to all spatial locations
        B, S, D = visual_emb.shape
        uncertainty_expanded = uncertainty_emb.expand(B, S, D)  # [B, spatial_dim, d_model]

        # Fuse visual and uncertainty
        memory = torch.cat([visual_emb, uncertainty_expanded], dim=-1)  # [B, spatial_dim, 2*d_model]
        memory = self.memory_fusion(memory)  # [B, spatial_dim, d_model]
        memory = memory.permute(1, 0, 2)  # [spatial_dim, B, d_model] - transformer format
        
        # Start with BOS token
        generated = torch.full(
            (batch_size, 1), 
            tokenizer.bos_token_id, 
            dtype=torch.long, 
            device=device
        )
        
        # ✅ NEW: Store attention weights for each step
        all_attention_weights = []
        
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for step in range(max_length - 1):
            # Get embeddings
            tgt_emb = self.token_embedding(generated) * math.sqrt(self.d_model)
            tgt_emb = tgt_emb.permute(1, 0, 2)
            tgt_emb = self.pos_encoder(tgt_emb)
            
            # Causal mask
            seq_len = generated.size(1)
            tgt_mask = self._generate_square_subsequent_mask(seq_len).to(device)
            
            # ✅ MODIFIED: Pass through decoder with attention capture
            output = tgt_emb
            step_attention = None
            
            for i, layer in enumerate(self.layers):
                # Get attention from last layer
                if i == len(self.layers) - 1:
                    layer_output = layer(output, memory, tgt_mask, return_attention=True)
                    if isinstance(layer_output, tuple):
                        output, step_attention = layer_output
                    else:
                        output = layer_output
                else:
                    output = layer(output, memory, tgt_mask)
            
            # ✅ Store attention for this step
            if step_attention is not None:
                # step_attention: [batch, nhead, seq_len, mem_len]
                # Take attention for last generated token: [:, :, -1, :]
                all_attention_weights.append(step_attention[:, :, -1, :].detach())
            
            # Get logits for last position
            output = output[-1, :, :]
            logits = self.output_proj(output)
            
            # Prevent EOS before min_length
            if step < min_length:
                logits[:, tokenizer.eos_token_id] = float('-inf')
            
            # Apply length penalty
            target_length = 40
            current_length = step + 1
            
            if current_length > target_length:
                penalty_factor = 1.0 + (current_length - target_length) * 0.1
                logits[:, tokenizer.eos_token_id] *= penalty_factor
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for token_id in set(generated[i].tolist()):
                        logits[i, token_id] /= repetition_penalty
            
            # Temperature scaling
            logits = logits / temperature
            
            # Top-p sampling
            probs = F.softmax(logits, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
            
            sorted_indices_to_remove = cumsum_probs > top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = False
            
            for i in range(batch_size):
                indices_to_remove = sorted_indices[i, sorted_indices_to_remove[i]]
                logits[i, indices_to_remove] = float('-inf')
            
            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Update finished status
            finished |= (next_token.squeeze(-1) == tokenizer.eos_token_id)
            
            # Stop if all finished
            if finished.all():
                break
            
            # Append
            generated = torch.cat([generated, next_token], dim=1)
        
        # ✅ Store all attention weights in the model
        if len(all_attention_weights) > 0:
            self.last_attention_weights = torch.stack(all_attention_weights, dim=2)
            # Shape: [batch, nhead, num_steps, mem_len]
        
        return generated


# ============================================================================
# CRITICAL: You also need to modify TransformerDecoderBlock
# ============================================================================

class TransformerDecoderBlock(nn.Module):
    """
    Single transformer decoder block with ATTENTION RETURN
    """
    
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None, return_attention=False):
        """
        Args:
            tgt: [seq_len, batch, d_model]
            memory: [mem_len, batch, d_model]
            tgt_mask: causal mask
            tgt_key_padding_mask: padding mask
            return_attention: ✅ NEW - whether to return cross-attention weights
        
        Returns:
            output: [seq_len, batch, d_model] OR
            (output, attention_weights) if return_attention=True
        """
        # Self attention
        tgt2, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # ✅ MODIFIED: Cross attention with optional weight return
        if return_attention:
            tgt2, cross_attn_weights = self.cross_attn(
                tgt, memory, memory,
                need_weights=True,  # ✅ Request attention weights
                average_attn_weights=False  # ✅ Keep all heads
            )
            tgt = tgt + self.dropout2(tgt2)
            tgt = self.norm2(tgt)
            
            # Feed forward
            tgt2 = self.feed_forward(tgt)
            tgt = tgt + self.dropout3(tgt2)
            tgt = self.norm3(tgt)
            
            return tgt, cross_attn_weights  # ✅ Return attention
        
        else:
            tgt2, _ = self.cross_attn(tgt, memory, memory)
            tgt = tgt + self.dropout2(tgt2)
            tgt = self.norm2(tgt)
            
            # Feed forward
            tgt2 = self.feed_forward(tgt)
            tgt = tgt + self.dropout3(tgt2)
            tgt = self.norm3(tgt)
            
            return tgt


def count_parameters(model):
    """Count trainable parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable