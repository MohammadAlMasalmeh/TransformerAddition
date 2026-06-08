class Tokenizer:
    def __init__(self):
        self.str_to_int = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "+": 10, "-": 11, "=": 12, " ":13}
        self.int_to_str = {0:"0", 1:"1", 2:"2", 3:"3", 4:"4", 5:"5", 6:"6", 7:"7", 8:"8", 9:"9", 10:"+", 11:"-", 12:"=", 13:" "}
    def encode(self, data):
        tokens = []
        for char in data:
            tokens.append(self.str_to_int[char])
        return tokens
    def decode(self, tokens):
        output = ""
        for token in tokens:
            output += self.int_to_str[token]
        return output 


# t = Tokenizer()
# encoded = t.encoder("285+-977=-0692")
# print(encoded)
# print(t.decoder(encoded))