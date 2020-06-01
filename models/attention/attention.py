import torch
import torch.nn.functional as F



class Attention(torch.nn.Module):

    def __init__(self, depth=32, enable_scale=True):
        super(Attention, self).__init__()
        self.depth = depth
        self.query_size = depth
        self.key_size = depth
        self.value_size = depth
        self.enable_scale = enable_scale
        self.conv_query = torch.nn.Conv1d(self.query_size, self.depth,
                                          kernel_size=1, stride=1, padding=0)
        self.conv_key = torch.nn.Conv1d(self.key_size, self.depth,
                                        kernel_size=1, stride=1, padding=0)
        self.conv_value = torch.nn.Conv1d(self.value_size, self.depth,
                                          kernel_size=1, stride=1, padding=0)

    def forward(self, x, memory):
        """ 
        Require:
            shape of x must be [batch, sentence_length, embedding_size].
            shape of memory must be [batch, sentence_length, embedding_size].
        """
        x = x.permute(0, 2, 1)
        memory = memory.permute(0, 2, 1)
        query = self.conv_query(x)
        if self.enable_scale:
            query *= self.depth ** -0.5
        key = self.conv_key(memory)
        value = self.conv_value(memory)
        relation = torch.bmm(query.permute(0, 2, 1), key)
        relation = F.softmax(relation, -1)
        y = torch.bmm(relation, value.permute(0, 2, 1))
        return y, relation



def unit_test():
    import numpy as np
    x = np.ones([64, 8, 32]).astype(np.float32)
    x = torch.from_numpy(x)
    memory = np.ones([64, 16, 32]).astype(np.float32)
    memory = torch.from_numpy(memory)
    attention = Attention(32)
    y, relation = attention(x, memory)
    print(f"output shape : {y.shape}")
    print(relation)


if __name__ == "__main__":
    unit_test()