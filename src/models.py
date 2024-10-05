from typing import Literal, Union, List
from transformers import AutoTokenizer, AutoModel, ViTModel, ViTFeatureExtractor
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
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
    
    def __init__(self, model_name: str = "openai/clip-vit-large-patch14", device: str = None) -> None:
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



class Generator(nn.Module):
    def __init__(self, img_size, latent_dim, dim):
        super(Generator, self).__init__()

        self.dim = dim
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.feature_sizes = (self.img_size[0] / 16, self.img_size[1] / 16)

        self.latent_to_features = nn.Sequential(
            nn.Linear(latent_dim, 8 * dim * self.feature_sizes[0] * self.feature_sizes[1]),
            nn.ReLU()
        )

        self.features_to_image = nn.Sequential(
            nn.ConvTranspose2d(8 * dim, 4 * dim, 4, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(4 * dim),
            nn.ConvTranspose2d(4 * dim, 2 * dim, 4, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(2 * dim),
            nn.ConvTranspose2d(2 * dim, dim, 4, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(dim),
            nn.ConvTranspose2d(dim, self.img_size[2], 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, input_data):
        # Map latent into appropriate size for transposed convolutions
        x = self.latent_to_features(input_data)
        # Reshape
        x = x.view(-1, 8 * self.dim, self.feature_sizes[0], self.feature_sizes[1])
        # Return generated image
        return self.features_to_image(x)

    def sample_latent(self, num_samples):
        return torch.randn((num_samples, self.latent_dim))


class Discriminator(nn.Module):
    def __init__(self, img_size, dim):
        """
        img_size : (int, int, int)
            Height and width must be powers of 2.  E.g. (32, 32, 1) or
            (64, 128, 3). Last number indicates number of channels, e.g. 1 for
            grayscale or 3 for RGB
        """
        super(Discriminator, self).__init__()

        self.img_size = img_size

        self.image_to_features = nn.Sequential(
            nn.Conv2d(self.img_size[2], dim, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim, 2 * dim, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(2 * dim, 4 * dim, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(4 * dim, 8 * dim, 4, 2, 1),
            nn.Sigmoid()
        )

        # 4 convolutions of stride 2, i.e. halving of size everytime
        # So output size will be 8 * (img_size / 2 ^ 4) * (img_size / 2 ^ 4)
        output_size = 8 * dim * (img_size[0] / 16) * (img_size[1] / 16)
        self.features_to_prob = nn.Sequential(
            nn.Linear(output_size, 1),
            nn.Sigmoid()
        )

    def forward(self, input_data):
        batch_size = input_data.size()[0]
        x = self.image_to_features(input_data)
        x = x.view(batch_size, -1)
        return self.features_to_prob(x)
    

class StackedGAN(nn.Module):
    def __init__(self, generator_dim: int, conditional_augmentation: bool):
        super(StackedGAN, self).__init__()
        self.generator_dim = generator_dim
        self.conditional_augmentation = conditional_augmentation
        self.build_module(conditional_augmentation)

    def build_module(self):
        if self.conditional_augmentation:
            self.conditional_augmentation_network = ConditionalAugmentationNetwork()


class ConditionalAugmentationNetwork(nn.Module):
    def __init__(self):
        super(ConditionalAugmentationNetwork, self).__init__()

    def forward(self, input_data):
        pass