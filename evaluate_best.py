
import torch
import torch.nn as nn
import math
import pickle
import os
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

# --- Load best config from tuning_results.pkl ---
def load_best_config(pkl_path="tuning_results.pkl"):
    with open(pkl_path, "rb") as f:
        best_result = pickle.load(f)
    # Ray Tune Result object: best_result.config is the dict
    if hasattr(best_result, "config"):
        return best_result.config
    elif isinstance(best_result, dict) and "config" in best_result:
        return best_result["config"]
    else:
        raise ValueError("Could not extract config from tuning_results.pkl")

BEST_CONFIG = load_best_config()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 50

# --- Model Components (copied from train_final.py for compatibility) ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.1)
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        Q = self.query_linear(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.key_linear(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.value_linear(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        if mask is not None: scores = scores.masked_fill(mask == 0, -1e9)
        attention_weights = torch.softmax(scores, dim=-1)
        attention_output = torch.matmul(self.dropout(attention_weights), V)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.out_linear(attention_output)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.linear2(self.dropout(self.relu(self.linear1(x))))

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
    def forward(self, x):
        mean = x.mean(-1, keepdim=True); std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model); self.norm2 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, mask=None):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, mask)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model); self.norm2 = LayerNorm(d_model); self.norm3 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.dropout(self.cross_attn(x, enc_out, enc_out, src_mask)))
        x = self.norm3(x + self.dropout(self.ffn(x)))
        return x

class Encoder(nn.Module):
    def __init__(self, input_vocab_size, d_model, num_layers, num_heads, d_ff, max_len, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(input_vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, mask=None):
        x = self.embed(x); x = self.pos_enc(x); x = self.dropout(x)
        for layer in self.layers: x = layer(x, mask)
        return x

class Decoder(nn.Module):
    def __init__(self, target_vocab_size, d_model, num_layers, num_heads, d_ff, max_len, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(target_vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        x = self.embed(x); x = self.pos_enc(x); x = self.dropout(x)
        for layer in self.layers: x = layer(x, enc_out, src_mask, tgt_mask)
        return x

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_layers=6, num_heads=8, d_ff=2048, max_len=100, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, d_model, num_layers, num_heads, d_ff, max_len, dropout)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_layers, num_heads, d_ff, max_len, dropout)
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
    def make_pad_mask(self, seq, pad_idx):
        return (seq != pad_idx).unsqueeze(1).unsqueeze(2)
    def make_subsequent_mask(self, size):
        return torch.tril(torch.ones((size, size))).bool().to(next(self.parameters()).device)
    def forward(self, src, tgt, src_pad_idx, tgt_pad_idx):
        src_mask = self.make_pad_mask(src, src_pad_idx)
        tgt_pad_mask = self.make_pad_mask(tgt, tgt_pad_idx)
        tgt_sub_mask = self.make_subsequent_mask(tgt.size(1))
        tgt_mask = tgt_pad_mask & tgt_sub_mask
        enc_out = self.encoder(src, src_mask)
        dec_out = self.decoder(tgt, enc_out, src_mask, tgt_mask)
        return self.fc_out(dec_out)

# --- Translation Logic ---
def translate_sentence(model, sentence, en_vocab, hi_vocab, max_len=50):
    model.eval()
    tokens = [en_vocab.stoi["<sos>"]] + en_vocab.numericalize(sentence)[:max_len-2] + [en_vocab.stoi["<eos>"]]
    src_tensor = torch.tensor(tokens).unsqueeze(0).to(DEVICE)
    tgt_tokens = [hi_vocab.stoi["<sos>"]]
    for _ in range(max_len):
        tgt_tensor = torch.tensor(tgt_tokens).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = model(src_tensor, tgt_tensor, en_vocab.stoi["<pad>"], hi_vocab.stoi["<pad>"])
        next_token = output[0, -1].argmax().item()
        tgt_tokens.append(next_token)
        if next_token == hi_vocab.stoi["<eos>"]: break
    return ' '.join([hi_vocab.itos[idx] for idx in tgt_tokens[1:-1]])

# --- Vocabulary (needed to load) ---
class Vocabulary:
    def __init__(self): self.stoi = {}; self.itos = {}
    def tokenize(self, sentence): return sentence.lower().strip().split()
    def numericalize(self, sentence): return [self.stoi.get(word, 3) for word in self.tokenize(sentence)]
    def __len__(self): return len(self.stoi)

# --- Main Evaluation ---
if __name__ == "__main__":
    with open("en_vocab.pkl", "rb") as f: en_vocab = pickle.load(f)
    with open("hi_vocab.pkl", "rb") as f: hi_vocab = pickle.load(f)

    model = Transformer(
        src_vocab_size=len(en_vocab),
        tgt_vocab_size=len(hi_vocab),
        d_model=BEST_CONFIG["d_model"],
        num_layers=BEST_CONFIG["num_layers"],
        num_heads=BEST_CONFIG["num_heads"],
        d_ff=BEST_CONFIG["d_ff"],
        max_len=MAX_LEN,
        dropout=BEST_CONFIG["dropout"],
    ).to(DEVICE)

    model.load_state_dict(torch.load("B23BB1032_ass_4_best_model.pth", map_location=DEVICE))
    
    val_dataset = [
        ("I love you.", "मैं तुमसे प्यार करता हूँ।"),
        ("How are you?", "आप कैसे हैं?"),
        ("You should sleep.", "आपको सोना चाहिए।"),
        ("Maybe Tom doesn't love you.", "टॉम शायद तुमसे प्यार नहीं करता है।"),
        ("Let me tell Tom.", "मुझे टॉम को बताने दीजिए।")
    ]

    references = []
    hypotheses = []
    smoothie = SmoothingFunction().method4

    for en, hi in val_dataset:
        pred = translate_sentence(model, en, en_vocab, hi_vocab)
        print(f"EN: {en}\nGT: {hi}\nPR: {pred}\n---")
        references.append([hi.split()])
        hypotheses.append(pred.split())

    score = corpus_bleu(references, hypotheses, smoothing_function=smoothie)
    print(f"FINAL BLEU SCORE: {score * 100:.2f}")
