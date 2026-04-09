import re
import json
from collections import Counter
from typing import List, Dict, Tuple
import pickle
import numpy as np
import torch

"""
Custom Medical Tokenizer for MIMIC-CXR Reports
Builds vocabulary from scratch and handles medical terminology
"""

import re
import pickle
from collections import Counter
from typing import List, Dict
import sentencepiece as spm  # pip install sentencepiece


class MedicalVocabulary:
    """
    Improved medical vocabulary with subword tokenization
    Based on approaches from R2Gen, CheXpert, and modern NLP papers
    """
    
    def __init__(
        self, 
        vocab_size: int = 5000,  # Reduced from 25k (following papers)
        use_subword: bool = True,
        preserve_case: bool = False,  # True for medical terms
        min_freq: int = 2
    ):
        self.vocab_size = vocab_size
        self.use_subword = use_subword
        self.preserve_case = preserve_case
        self.min_freq = min_freq
        
        # Special tokens (minimal set)
        self.pad_token = '<pad>'
        self.unk_token = '<unk>'
        self.bos_token = '<s>'
        self.eos_token = '</s>'
        
        # Initialize
        self.word2idx = {}
        self.idx2word = {}
        self.word_freq = Counter()
        self.sp_model = None  # SentencePiece model
        
        # Medical term dictionary (high-priority terms)
        self.medical_terms = self._load_medical_terms()
    
    def _load_medical_terms(self) -> set:
        """
        Load critical medical terms that should ALWAYS be in vocab
        These bypass min_freq requirement
        """
        # Core anatomical terms
        anatomy = {
            'lung', 'lungs', 'cardiac', 'heart', 'mediastinum', 'pleural',
            'diaphragm', 'trachea', 'aorta', 'hilum', 'hilar', 'apex',
            'base', 'lobe', 'lobes', 'thorax', 'chest', 'rib', 'ribs',
            'clavicle', 'spine', 'vertebra', 'costophrenic', 'hemidiaphragm'
        }
        
        # Common findings
        findings = {
            'consolidation', 'effusion', 'pneumothorax', 'atelectasis',
            'opacity', 'opacities', 'edema', 'infiltrate', 'infiltrates',
            'pneumonia', 'cardiomegaly', 'nodule', 'nodules', 'mass',
            'lesion', 'lesions', 'fracture', 'fractures', 'air',
            'fluid', 'thickening', 'calcification', 'emphysema'
        }
        
        # Descriptors
        descriptors = {
            'normal', 'abnormal', 'clear', 'enlarged', 'small', 'large',
            'mild', 'moderate', 'severe', 'minimal', 'stable', 'unchanged',
            'increased', 'decreased', 'bilateral', 'unilateral', 'focal',
            'diffuse', 'left', 'right', 'upper', 'lower', 'middle',
            'basilar', 'apical', 'peripheral', 'central', 'lateral',
            'medial', 'anterior', 'posterior'
        }
        
        # Negations (critical for accuracy!)
        negations = {
            'no', 'not', 'without', 'absent', 'negative', 'none'
        }
        
        # Combine all
        return anatomy | findings | descriptors | negations
    
    def build_from_reports(self, reports: List[str], use_sentencepiece: bool = False):
        """
        Build vocabulary from reports
        
        Args:
            reports: List of clean findings texts
            use_sentencepiece: Use SentencePiece for subword tokenization
        """
        print(f"Building vocabulary from {len(reports)} reports...")
        print(f"  Vocab size: {self.vocab_size}")
        print(f"  Subword tokenization: {self.use_subword}")
        print(f"  Preserve case: {self.preserve_case}")
        
        if use_sentencepiece and self.use_subword:
            self._build_sentencepiece(reports)
        else:
            self._build_word_level(reports)
        
        return self
    
    def _build_word_level(self, reports: List[str]):
        """Build word-level vocabulary (original approach, improved)"""
        
        # Tokenize and count
        for report in reports:
            tokens = self._tokenize_medical(report)
            self.word_freq.update(tokens)
        
        # Start with special tokens
        vocab_list = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        
        # Add medical terms (bypass min_freq)
        medical_in_data = []
        for term in self.medical_terms:
            term_lower = term.lower()
            if term_lower in self.word_freq:
                medical_in_data.append(term_lower)
        
        vocab_list.extend(sorted(medical_in_data))
        
        # Add frequent words
        frequent_words = [
            word for word, freq in self.word_freq.items()
            if freq >= self.min_freq and word not in medical_in_data
        ]
        
        # Sort by frequency
        frequent_words = sorted(
            frequent_words,
            key=lambda w: self.word_freq[w],
            reverse=True
        )
        
        # Limit size
        remaining_space = self.vocab_size - len(vocab_list)
        if len(frequent_words) > remaining_space:
            frequent_words = frequent_words[:remaining_space]
        
        vocab_list.extend(frequent_words)
        
        # Build mappings
        self.word2idx = {word: idx for idx, word in enumerate(vocab_list)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        
        print(f"✅ Vocabulary built: {len(self.word2idx)} tokens")
        print(f"   Special tokens: 4")
        print(f"   Protected medical terms: {len(medical_in_data)}")
        print(f"   Frequent words: {len(frequent_words)}")
        print(f"   Coverage: {self._compute_coverage(reports):.2%}")
    
    def _build_sentencepiece(self, reports: List[str]):
        """Build SentencePiece model (subword tokenization)"""
        import tempfile
        import os
        
        # Write reports to temp file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            for report in reports:
                f.write(report + '\n')
            temp_file = f.name
        
        # Train SentencePiece
        model_prefix = tempfile.mktemp()
        
        spm.SentencePieceTrainer.train(
            input=temp_file,
            model_prefix=model_prefix,
            vocab_size=self.vocab_size,
            character_coverage=0.9995,
            model_type='bpe',  # or 'unigram'
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            user_defined_symbols=list(self.medical_terms)  # Protect medical terms
        )
        
        # Load model
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.load(model_prefix + '.model')
        
        # Build mappings
        self.word2idx = {
            self.sp_model.id_to_piece(i): i 
            for i in range(self.sp_model.get_piece_size())
        }
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        
        # Cleanup
        os.unlink(temp_file)
        os.unlink(model_prefix + '.model')
        os.unlink(model_prefix + '.vocab')
        
        print(f"✅ SentencePiece vocabulary built: {len(self.word2idx)} tokens")
        print(f"   Coverage: {self._compute_coverage(reports):.2%}")
    
    def _tokenize_medical(self, text: str) -> List[str]:
        """
        Improved medical tokenization
        
        Key improvements:
        - Optional case preservation
        - Better handling of compounds
        - Hyphenated terms kept together
        """
        if not self.preserve_case:
            text = text.lower()
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        # Protect medical compounds with hyphens
        # "well-defined" → "well_hyphen_defined" temporarily
        text = re.sub(r'(\w+)-(\w+)', r'\1_hyphen_\2', text)
        
        # Protect measurements
        # "5.2 cm" → "5.2cm"
        text = re.sub(r'(\d+\.?\d*)\s*(cm|mm|ml|cc|mg|kg)', r'\1\2', text)
        
        # Protect abbreviations
        # "p.a." → "p_dot_a_dot_"
        text = re.sub(r'([a-z])\.([a-z])\.', r'\1_dot_\2_dot_', text, flags=re.IGNORECASE)
        
        # Tokenize: words, numbers, and punctuation
        tokens = re.findall(r'\w+|[.,!?;:()\[\]]', text)
        
        # Restore hyphens and dots
        tokens = [
            t.replace('_hyphen_', '-').replace('_dot_', '.')
            for t in tokens
        ]
        
        return tokens
    
    def _compute_coverage(self, reports: List[str]) -> float:
        """Compute vocabulary coverage (% of tokens in vocab)"""
        total_tokens = 0
        covered_tokens = 0
        
        for report in reports[:1000]:  # Sample for speed
            tokens = self._tokenize_medical(report)
            total_tokens += len(tokens)
            
            for token in tokens:
                if token in self.word2idx:
                    covered_tokens += 1
        
        return covered_tokens / total_tokens if total_tokens > 0 else 0.0
    
    def encode(self, text: str) -> List[int]:
        """Convert text to token IDs"""
        if self.sp_model:
            # SentencePiece encoding
            return self.sp_model.encode(text)
        
        # Word-level encoding
        tokens = self._tokenize_medical(text)
        
        token_ids = [self.word2idx[self.bos_token]]
        
        for token in tokens:
            # Try exact match first
            if token in self.word2idx:
                token_ids.append(self.word2idx[token])
            # Try lowercase (if preserve_case=True)
            elif self.preserve_case and token.lower() in self.word2idx:
                token_ids.append(self.word2idx[token.lower()])
            else:
                token_ids.append(self.word2idx[self.unk_token])
        
        token_ids.append(self.word2idx[self.eos_token])
        
        return token_ids
    
    def decode(self, token_ids: List[int], skip_special: bool = True) -> str:
        """Convert token IDs back to text"""
        if self.sp_model:
            return self.sp_model.decode(token_ids)
        
        tokens = []
        
        for idx in token_ids:
            word = self.idx2word.get(idx, self.unk_token)
            
            # Skip special tokens
            if skip_special and word in [
                self.pad_token, self.unk_token, 
                self.bos_token, self.eos_token
            ]:
                continue
            
            tokens.append(word)
        
        # Join tokens
        text = ' '.join(tokens)
        
        # Fix punctuation spacing
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        text = re.sub(r'(\()\s+', r'\1', text)
        text = re.sub(r'\s+(\))', r'\1', text)
        
        return text.strip()
    
    def save(self, path: str):
        """Save vocabulary"""
        data = {
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
            'word_freq': dict(self.word_freq),
            'config': {
                'vocab_size': self.vocab_size,
                'use_subword': self.use_subword,
                'preserve_case': self.preserve_case,
                'min_freq': self.min_freq
            }
        }
        
        if self.sp_model:
            # Save SentencePiece model separately
            sp_path = path.replace('.pkl', '.model')
            self.sp_model.save(sp_path)
            data['sp_model_path'] = sp_path
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"✅ Vocabulary saved to {path}")
    
    @classmethod
    def load(cls, path: str):
        """Load vocabulary"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        config = data['config']
        vocab = cls(
            vocab_size=config['vocab_size'],
            use_subword=config['use_subword'],
            preserve_case=config['preserve_case'],
            min_freq=config['min_freq']
        )
        
        vocab.word2idx = data['word2idx']
        vocab.idx2word = data['idx2word']
        vocab.word_freq = Counter(data['word_freq'])
        
        # Load SentencePiece if exists
        if 'sp_model_path' in data:
            vocab.sp_model = spm.SentencePieceProcessor()
            vocab.sp_model.load(data['sp_model_path'])
        
        print(f"✅ Vocabulary loaded: {len(vocab.word2idx)} tokens")
        return vocab
    
    def __len__(self):
        return len(self.word2idx)
    
    @property
    def pad_token_id(self):
        return self.word2idx[self.pad_token]
    
    @property
    def unk_token_id(self):
        return self.word2idx[self.unk_token]
    
    @property
    def bos_token_id(self):
        return self.word2idx[self.bos_token]
    
    @property
    def eos_token_id(self):
        return self.word2idx[self.eos_token]


class MedicalTokenizer:
    """High-level tokenizer with batching support"""
    
    def __init__(self, vocab: MedicalVocabulary, max_length: int = 128):
        self.vocab = vocab
        self.max_length = max_length
    
    def __call__(self, texts, padding=True, truncation=True, 
                 return_tensors=None, max_length=None):
        """Tokenize with batching support (HuggingFace-like interface)"""
        import torch
        
        if isinstance(texts, str):
            texts = [texts]
        
        max_len = max_length or self.max_length
        
        # Encode all texts
        encoded = [self.vocab.encode(text) for text in texts]
        
        # Truncate
        if truncation:
            encoded = [ids[:max_len] for ids in encoded]
        
        # Pad
        if padding:
            max_seq_len = max(len(ids) for ids in encoded)
            max_seq_len = min(max_seq_len, max_len)
            
            encoded = [
                ids + [self.vocab.pad_token_id] * (max_seq_len - len(ids))
                for ids in encoded
            ]
        
        # Convert to tensors
        if return_tensors == 'pt':
            encoded = torch.tensor(encoded, dtype=torch.long)
            return {'input_ids': encoded}
        
        return encoded
    
    def encode(self, text, max_length=None):
        """Encode single text to token IDs"""
        max_len = max_length or self.max_length
        token_ids = self.vocab.encode(text)
        
        # Truncate if needed
        if len(token_ids) > max_len:
            token_ids = token_ids[:max_len]
        
        return torch.tensor(token_ids, dtype=torch.long)
    
    def decode(self, token_ids, skip_special_tokens=True):
        """Decode tokens back to text"""
        import torch
        
        if torch.is_tensor(token_ids):
            token_ids = token_ids.cpu().tolist()
        
        if isinstance(token_ids[0], list):
            # Batch decoding
            return [
                self.vocab.decode(ids, skip_special=skip_special_tokens)
                for ids in token_ids
            ]
        else:
            # Single sequence
            return self.vocab.decode(token_ids, skip_special=skip_special_tokens)
    
    @property
    def pad_token_id(self):
        return self.vocab.pad_token_id
    
    @property
    def eos_token_id(self):
        return self.vocab.eos_token_id
    
    @property
    def bos_token_id(self):
        return self.vocab.bos_token_id


def build_mimic_vocabulary(train_df, vocab_save_path, min_freq=2, max_vocab=5000):
    """Build vocabulary from MIMIC-CXR training data"""
    
    # ✅ Use FINDINGS_ONLY column (not Report or Report_Clean)
    if 'Findings_Only' in train_df.columns:
        reports = train_df['Findings_Only'].dropna().tolist()
    elif 'Report_Clean' in train_df.columns:
        reports = train_df['Report_Clean'].dropna().tolist()
    else:
        reports = train_df['Report'].dropna().tolist()
    
    # Remove duplicates
    reports = list(set(reports))
    
    print(f"Building vocabulary from {len(reports)} unique reports...")
    
    # Build vocabulary
    vocab = MedicalVocabulary(min_freq=min_freq, max_vocab_size=max_vocab)
    vocab.build_from_reports(reports)
    
    # Save
    vocab.save(vocab_save_path)
    
    print(f"\nVocabulary Statistics:")
    print(f"  Total tokens: {len(vocab)}")
    print(f"  Min frequency threshold: {min_freq}")
    print(f"  Max vocab size: {max_vocab}")
    print(f"  Unique words in data: {len(vocab.word_freq)}")
    
    return vocab


def build_and_save_vocabulary(
    train_df,
    save_path,
    findings_column='Findings_Clean',
    min_freq=2,
    verbose=True
):
    """Simplified version that works with your existing class"""
    import os
    from Vocabulary.vocabulary import MedicalVocabulary
    
    print("="*80)
    print("BUILDING MEDICAL VOCABULARY")
    print("="*80)
    
    # Create directory
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Get reports
    reports = train_df[findings_column].dropna().astype(str).tolist()
    reports = [r.strip() for r in reports if len(r.strip()) > 10]
    unique_reports = list(set(reports))
    
    print(f"\nData: {len(reports)} reports, {len(unique_reports)} unique")
    
    # Build vocabulary (using only parameters your class accepts)
    vocab = MedicalVocabulary(min_freq=min_freq)  # ← Only min_freq
    vocab.build_from_reports(unique_reports)
    
    # Save
    vocab.save(save_path)
    print(f"✅ Saved to: {save_path}")
    
    if verbose:
        print(f"\nVocabulary size: {len(vocab)} tokens")
        test = "The lungs are clear."
        print(f"Test encode: {vocab.encode(test)[:5]}...")
    
    return vocab