# inference.py
import torch
from data_generator import format_num
from tokenizer import Tokenizer
from model import TransformerModel
from normalizer import normalize

def load_model():
    tokenizer = Tokenizer()
    model = TransformerModel(
        vocab_size=14,
        embed_dim=64,
        num_heads=4,
        num_layers=3,
        max_seq_len=20
    )
    model.load_state_dict(torch.load("model.pt"))
    model.eval()
    return model, tokenizer

def run_inference(model, tokenizer, user_input):
    val1, val2 = normalize(user_input)
    canonical = format_num(val1, 3) + " " + format_num(val2, 3) + " = "
    tokens = tokenizer.encode(canonical)
    current_input = torch.tensor(tokens)

    with torch.no_grad():
        for _ in range(5):
            logits = model(current_input.unsqueeze(0))
            next_token = torch.argmax(logits[0, -1]).item()
            if next_token == 13:
                break
            tokens.append(next_token)
            current_input = torch.tensor(tokens)

    return val1, val2, tokens