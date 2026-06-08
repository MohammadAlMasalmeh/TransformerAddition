# main.py
import os
import sys

if not os.path.exists("model.pt") or "--retrain" in sys.argv:
    import train

from inference import load_model, run_inference

model, tokenizer = load_model()
print("model loaded. ask for any three digit addition. type quit to exit.")

while True:
    user_input = input("> ")
    if user_input.strip().lower() == "quit":
        break
    try:
        val1, val2, tokens = run_inference(model, tokenizer, user_input)
        answer = int(tokenizer.decode(tokens).split('=')[1].strip())
        print(f"{val1} + {val2} = {answer}")
    except Exception as e:
        print(f"error: {e}")