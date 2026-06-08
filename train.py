import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_generator import generate_dataset
from tokenizer import Tokenizer
from dataset import AddDataset
from model import TransformerModel

data = generate_dataset(50000)
tokenizer = Tokenizer()
train_dataset = AddDataset(data[:49000], tokenizer)
test_dataset = AddDataset(data[49000:], tokenizer)
dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)



model = TransformerModel(
    vocab_size=14,
    embed_dim=64,
    num_heads=4,
    num_layers=3,
    max_seq_len=20
)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

epochs = 20

def evaluate(model, dataset):
    model.eval()
    correct = 0
    total = len(dataset)

    with torch.no_grad():
        for i in range(total):
            x, _ = dataset[i]
            eq_pos = (x == 12).nonzero()[0].item()
            input_tokens = x[:eq_pos + 2]
            answer_tokens = x[eq_pos + 2:]
            answer_tokens = answer_tokens[answer_tokens != 13] 
            generated = []
            current_input = input_tokens.clone()
            for j in range(len(answer_tokens)):
                logits = model(current_input.unsqueeze(0))
                next_token = torch.argmax(logits[0, -1]).item()
                generated.append(next_token)
                current_input = torch.cat([current_input, torch.tensor([next_token])])
            if generated == answer_tokens.tolist():
                correct += 1

    accuracy = correct / total
    # print(f"input: {tokenizer.decode(x.tolist())}")
    # print(f"predicted: {tokenizer.decode(generated)}")
    # print(f"correct: {tokenizer.decode(answer_tokens.tolist())}")
    model.train()
    return accuracy
            


best_acc = 0
for epoch in range(epochs):
    total_loss = 0
    for x, y in dataloader:
        logits = model(x)
        loss = criterion(logits.view(-1, 14), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    acc = evaluate(model, test_dataset)
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "model.pt")
    print(f"epoch {epoch} | loss {avg_loss:.4f} | accuracy {acc:.4f}")
