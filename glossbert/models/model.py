"""
GlossBERT model implementation for Word Sense Disambiguation.
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel
from typing import Dict, Optional, Tuple, Union

class GlossBERT(BertPreTrainedModel):
    """
    GlossBERT model for Word Sense Disambiguation.
    Fine-tunes BERT for sentence-pair classification to determine if a gloss matches a context.
    """
    
    def __init__(self, config):
        """
        Initialize GlossBERT model.
        
        Args:
            config: BERT configuration
        """
        super().__init__(config)
        
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 2)  # Binary classification
        
        # Initialize weights
        self.init_weights()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Forward pass of the GlossBERT model.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            token_type_ids: Token type IDs
            position_ids: Position IDs
            head_mask: Head mask
            inputs_embeds: Input embeddings
            labels: Labels for loss computation
            output_attentions: Whether to output attentions
            output_hidden_states: Whether to output hidden states
            return_dict: Whether to return a dictionary
            
        Returns:
            Model outputs including loss and logits
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Pass inputs through BERT
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # Get the [CLS] token representation
        pooled_output = outputs[1]
        
        # Apply dropout and classification layer
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))
        
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': outputs.hidden_states,
            'attentions': outputs.attentions,
        }

def load_glossbert_model(model_path: Optional[str] = None):
    """
    Load a pre-trained or fine-tuned GlossBERT model.
    
    Args:
        model_path: Path to the fine-tuned model, or None to load pre-trained BERT
        
    Returns:
        Loaded model and tokenizer
    """
    from transformers import BertTokenizer, BertConfig
    
    if model_path:
        # Load fine-tuned model
        config = BertConfig.from_pretrained(model_path)
        model = GlossBERT.from_pretrained(model_path, config=config)
        tokenizer = BertTokenizer.from_pretrained(model_path)
    else:
        # Load pre-trained BERT
        config = BertConfig.from_pretrained('bert-base-uncased')
        model = GlossBERT.from_pretrained('bert-base-uncased', config=config)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    return model, tokenizer
