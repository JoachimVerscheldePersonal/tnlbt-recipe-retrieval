from transformers import AutoTokenizer, AutoModel
import torch

class RecipeIngredientEncoder():

    def __init__(self) -> None:
        pass  

class RecipeInstructionsEncoder():

    def __init__(self) -> None:
        pass

class RecipeTitleEncoder():

    def  __init__(self) -> None:
        pass

class RecipeImageEncoder():

    def __init__(self) -> None:
        pass

class HierarchicalTransformer():

    def __init__(self, model_name: str = "kiddothe2b/hierarchical-transformer-base-4096") -> None:
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)


    def forward(self, text:str)
        # Tokenize the input text and convert to PyTorch tensors
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=4096)

        # Get embeddings from the model
        with torch.no_grad():
            outputs = self.model(**inputs)

        # The outputs contain several elements, but we are interested in the last hidden states
        # The last_hidden_state is the output embedding for each token in the input
        last_hidden_states = outputs.last_hidden_state

        # You can aggregate these token embeddings (e.g., using mean pooling) to create a single embedding for the whole sentence
        sentence_embedding = torch.mean(last_hidden_states, dim=1)

        # Print the shape of the embeddings (for a single sentence, this would be 1 x hidden_size)
        print(sentence_embedding.shape)

        # To access the embeddings, you can convert them to a numpy array if needed
        embedding_array = sentence_embedding.numpy()
        print(embedding_array)
