import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import torch
from torch import nn
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
from reports_LLM_Components.reports_generation_components import StructuredDiagnosticOutput
class InteractiveDiagnosticDialogue:
    """Enable conversational interaction about the diagnosis"""
    def __init__(self, model, llm_generator):
        self.model = model
        self.llm_generator = llm_generator
        self.conversation_history = []
        self.current_diagnosis = None
        
    def process_image_and_start_dialogue(self, image, device):
        """Process image and initialize dialogue"""
        # Get diagnosis from your Bayesian model
        with torch.no_grad():
            outputs = self.model({'images': image.unsqueeze(0)}, device, 
                                generate_report=True)
        
        self.current_diagnosis = outputs
        initial_report = outputs.get('llm_reports', ['No report generated'])[0]
        
        self.conversation_history.append({
            'role': 'assistant',
            'content': initial_report
        })
        
        return initial_report
    
    def ask_question(self, question: str):
        """Handle follow-up questions about the diagnosis"""
        if not self.current_diagnosis:
            return "Please provide an image first for diagnosis."
        
        # Create context from current diagnosis
        structured = StructuredDiagnosticOutput(
            self.current_diagnosis['disease_logits'][0],
            {k: v[0] for k, v in self.current_diagnosis['class_uncertainties'].items()},
            self.current_diagnosis['consistency_score'][0]
        )
        
        # Build prompt with conversation history
        prompt = f"""Based on the following chest X-ray analysis:

{structured.to_prompt()}

Previous conversation:
{self._format_conversation()}

User question: {question}

Provide a medically accurate response that appropriately reflects the uncertainty levels in the diagnosis:"""
        
        # Generate response
        response = self.llm_generator.llm.generate(
            self.llm_generator.tokenizer(prompt, return_tensors="pt")['input_ids'].to(device),
            max_length=256,
            temperature=0.7
        )
        
        response_text = self.llm_generator.tokenizer.decode(response[0], skip_special_tokens=True)
        
        # Update conversation history
        self.conversation_history.append({'role': 'user', 'content': question})
        self.conversation_history.append({'role': 'assistant', 'content': response_text})
        
        return response_text
    
    def _format_conversation(self):
        """Format conversation history for prompt"""
        return "\n".join([f"{msg['role']}: {msg['content']}" for msg in self.conversation_history[-4:]])
