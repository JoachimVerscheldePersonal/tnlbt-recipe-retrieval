from transformers import AutoTokenizer, AutoModel
import torch
import torch
from transformers import AutoTokenizer, AutoModel

class HierarchicalTransformerEncoder:
    def __init__(self, model_name="kiddothe2b/hierarchical-transformer-base-4096", device=None):
        """
        Initialize the encoder with the specified model and tokenizer.
        Args:
            model_name (str): The Hugging Face model name.
            device (str): 'cpu' or 'cuda' (GPU). If None, defaults to available GPU if present.
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
    
    def encode(self, text, pooling="mean", max_length=4096):
        """
        Tokenize input text and return the sentence embedding.
        
        Args:
            text (str): The input text to encode.
            pooling (str): Pooling strategy - 'mean', 'cls'. Defaults to 'mean'.
            max_length (int): Maximum sequence length. Defaults to 4096.
            
        Returns:
            numpy.ndarray: The embedding for the input text.
        """
        # Tokenize input text
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, 
                                padding="max_length", max_length=max_length)
        
        # Move input tensors to the specified device (GPU/CPU)
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        
        # Get model outputs without computing gradients
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Retrieve the last hidden states (token embeddings)
        last_hidden_states = outputs.last_hidden_state
        
        # Apply pooling strategy to get a single vector for the input text
        if pooling == "mean":
            # Mean pooling over all tokens to get the sentence embedding
            sentence_embedding = torch.mean(last_hidden_states, dim=1)
        elif pooling == "cls":
            # Use the [CLS] token embedding (assuming it's available as the first token)
            sentence_embedding = last_hidden_states[:, 0, :]
        else:
            raise ValueError("Unsupported pooling type. Choose either 'mean' or 'cls'.")
        
        # Move the embedding back to CPU and convert to numpy array
        return sentence_embedding.cpu().numpy()

