class CharTokenizer():
    def __init__(self, intext: str):
        self.charset = set(intext)
        sorted_charset = sorted(list(self.charset))
        self.stoi = {c:i for i,c in enumerate(sorted_charset)}
        self.itos = {i:c for i,c in enumerate(sorted_charset)}
        self.n_vocab = len(self.charset)
    
    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]
    
    def decode(self, l: list[int]) -> str:
        return ''.join(self.itos[i] for i in l)