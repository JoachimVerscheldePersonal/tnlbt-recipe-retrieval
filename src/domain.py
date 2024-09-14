from torch.nn import TransformerEncoderLayer
from torch import rand

class RecipeEncoder():

    def __init__(self, embedding_size = 512, attention_heads = 8):
        self.encoder_layer = TransformerEncoderLayer(d_model=embedding_size, n_head=attention_heads)
        src = rand(10,32,embedding_size)
        out = self.encoder_layer(src)