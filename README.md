# Transformer Addition

A transformer trained from scratch to do 3-digit addition.

## How it works

Generates 50k addition problems (e.g. `123 + 456 = 579`), tokenizes them as sequences, and trains a small transformer encoder to predict the answer token by token.

## Run it

```bash
python main.py
```

Type any 3 digit addition problem and it'll answer. `quit` to exit.

## Retrain

```bash
python main.py --retrain
```

## Stack

- PyTorch
- Custom tokenizer + dataset
- 3-layer transformer encoder, 4 heads, 64 embed dim
