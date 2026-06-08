from data_generator import generate_dataset
from tokenizer import Tokenizer
from dataset import AddDataset
from model import TransformerModel
from torch.utils.data import DataLoader

# data
data = generate_dataset(1000)
tokenizer = Tokenizer()
dataset = AddDataset(data, tokenizer)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# model
model = TransformerModel(
    vocab_size=14,
    embed_dim=64,
    num_heads=4,
    num_layers=3,
    max_seq_len=20
)

# test single example
x, y = dataset[0]
print("input tokens:", x)
print("target tokens:", y)

# test model forward pass
x_batch = x.unsqueeze(0)  # add batch dimension [1, seq_len]
logits = model(x_batch)
print("logits shape:", logits.shape)  # should be [1, 14, 13]