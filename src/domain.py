from typing import Literal, Union, List
from transformers import AutoTokenizer, AutoModel, ViTModel, ViTFeatureExtractor
from PIL import Image
import numpy as np
import torch
import requests

class HierarchicalTransformerEncoder:
    def __init__(self, model_name: str = "kiddothe2b/hierarchical-transformer-base-4096", device: str = None):
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
    
    def encode(self, text: str, pooling: Literal["mean", "cls"] = "mean", max_length: int = 4096) -> np.ndarray:
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


class VisualTransformerEncoder: 
    
    def __init__(self, model_name: str = "google/vit-base-patch16-224", device: str = None) -> None:
        """
        Initialize the encoder with the specified ViT model and feature extractor.
        Args:
            model_name (str): The Hugging Face ViT model name.
            device (str): 'cpu' or 'cuda' (GPU). If None, defaults to available GPU if present.
        """

        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

        # Load the feature extractor and model
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
        self.model = ViTModel.from_pretrained(model_name).to(self.device)

    def encode(self, images: List[Union[Image.Image, str]], pooling: Literal["mean", "cls"] = "mean") -> np.ndarray:
        """
        Encode an input image into a feature embedding.
        
        Args:
            images (List[PIL.Image or str]): A list of images (PIL Images or URLs/local paths).
            pooling (str): Pooling strategy - 'mean', 'cls'. Defaults to 'mean'.
            
        Returns:
            numpy.ndarray: The embedding for the input image.
        """

        if not images:
            raise ValueError("List of images cannot be empty.")

        # Load image if it's a path or URL
        if isinstance(images[0], str):
            images = [Image.open(requests.get(img, stream=True).raw) if img.startswith('http') else Image.open(img) for img in images]


         # Preprocess the image using the feature extractor
        inputs = self.feature_extractor(images=images, return_tensors="pt")

        # Move input tensors to the specified device (GPU/CPU)
        inputs = {key: val.to(self.device) for key, val in inputs.items()}

        # Get model outputs without computing gradients
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Retrieve the last hidden states (token embeddings)
        last_hidden_states = outputs.last_hidden_state
        
        # Apply pooling strategy to get a single vector for each image
        if pooling == "mean":
            # Mean pooling over all tokens for each image in the batch
            image_embeddings = torch.mean(last_hidden_states, dim=1)
        elif pooling == "cls":
            # Use the [CLS] token embedding (first token) for each image in the batch
            image_embeddings = last_hidden_states[:, 0, :]
        else:
            raise ValueError("Unsupported pooling type. Choose either 'mean' or 'cls'.")
        
        # Move the embeddings back to CPU and convert to numpy array
        return image_embeddings.cpu().numpy()

