import torch
import torch.nn.functional as F



class SkipGram(torch.nn.Module):

    def __init__(self, num_vocabulary, embedding_dim=64):
        super(SkipGram, self).__init__()
        self.num_vocabulary = num_vocabulary
        self.embedding_dim = embedding_dim
        self.encode_embedding = torch.nn.Embedding(self.num_vocabulary,
                                                   self.embedding_dim)
        self.decode_embedding = torch.nn.Embedding(self.num_vocabulary,
                                                   self.embedding_dim)

    def forward(self, word, context):
        h_word = self.encode_embedding(word)
        h_context = self.decode_embedding(context)
        return h_word, h_context

        

        
