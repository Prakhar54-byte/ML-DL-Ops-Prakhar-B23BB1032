import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import math
import os
import pickle
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import ray
from ray import tune
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler
import ray.train
import sys
sys.stdout = open('/home/raid/vishal_kishore/assignment_04/tuning.log', 'a')
sys.stderr = sys.stdout

from collections import Counter

# --- Vocabulary and Dataset ---

class Vocabulary:
    def __init__(self, freq_threshold=2):
        self.freq_threshold = freq_threshold
        self.itos = {0: "<pad>", 1: "<sos>", 2: "<eos>", 3: "<unk>"}
        self.stoi = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
        self.idx = 4

    def build_vocab(self, sentence_list):
        frequencies = Counter()
        for sentence in sentence_list:
            for word in self.tokenize(sentence):
                frequencies[word] += 1

        for word, freq in frequencies.items():
            if freq >= self.freq_threshold:
                self.stoi[word] = self.idx
                self.itos[self.idx] = word
                self.idx += 1

    def tokenize(self, sentence):
        return sentence.lower().strip().split()

    def numericalize(self, sentence):
        tokens = self.tokenize(sentence)
        return [self.stoi.get(token, self.stoi["<unk>"]) for token in tokens]

    def __len__(self):
        return len(self.stoi)

    def __getitem__(self, token):
        return self.stoi.get(token, self.stoi["<unk>"])

def encode_sentence(sentence, vocab, max_len=50):
    tokens = [vocab.stoi["<sos>"]] + vocab.numericalize(sentence)[:max_len-2] + [vocab.stoi["<eos>"]]
    return tokens + [vocab.stoi["<pad>"]] * (max_len - len(tokens))

class TranslationDataset(Dataset):
    def __init__(self, df, en_vocab, hi_vocab, max_len=50):
        self.en_sentences = df["en"].tolist()
        self.hi_sentences = df["hi"].tolist()
        self.en_vocab = en_vocab
        self.hi_vocab = hi_vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.en_sentences)

    def __getitem__(self, idx):
        src = encode_sentence(self.en_sentences[idx], self.en_vocab, self.max_len)
        tgt = encode_sentence(self.hi_sentences[idx], self.hi_vocab, self.max_len)
        return torch.tensor(src), torch.tensor(tgt)

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_batch = torch.stack(src_batch)
    tgt_batch = torch.stack(tgt_batch)
    tgt_input = tgt_batch[:, :-1]
    tgt_output = tgt_batch[:, 1:]
    return src_batch, tgt_input, tgt_output

# --- Transformer Model ---

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
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
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
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
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
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
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
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
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
        x = self.embed(x)
        x = self.pos_enc(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x

class Decoder(nn.Module):
    def __init__(self, target_vocab_size, d_model, num_layers, num_heads, d_ff, max_len, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(target_vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        x = self.embed(x)
        x = self.pos_enc(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, enc_out, src_mask, tgt_mask)
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

# --- Training and Tuning ---

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 50

# Global dataframe for trials
df_global = None

def get_data():
    global df_global
    if df_global is None:
        df_global = pd.read_csv('/home/raid/vishal_kishore/assignment_04/English-Hindi.tsv', sep='\t', header=None, names=["id1", "en", "id2", "hi"])
        df_global = df_global[["en", "hi"]].dropna().reset_index(drop=True)
        df_global = df_global.head(10000)
    return df_global

# Load shared vocabs
with open("/home/raid/vishal_kishore/assignment_04/en_vocab.pkl", "rb") as f:
    en_vocab = pickle.load(f)
with open("/home/raid/vishal_kishore/assignment_04/hi_vocab.pkl", "rb") as f:
    hi_vocab = pickle.load(f)

SRC_PAD_IDX = en_vocab["<pad>"]
TGT_PAD_IDX = hi_vocab["<pad>"]

def train_tune(config):
    df = get_data()
    dataset = TranslationDataset(df, en_vocab, hi_vocab, max_len=MAX_LEN)
    train_loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_fn)

    model = Transformer(
        src_vocab_size=len(en_vocab),
        tgt_vocab_size=len(hi_vocab),
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        max_len=MAX_LEN,
        dropout=config["dropout"],
    ).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config.get("weight_decay", 1e-4))
    criterion = nn.CrossEntropyLoss(ignore_index=TGT_PAD_IDX)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.get("epochs", 20))

    epochs = config.get("epochs", 20)
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for src, tgt_input, tgt_output in train_loader:
            src, tgt_input, tgt_output = src.to(DEVICE), tgt_input.to(DEVICE), tgt_output.to(DEVICE)
            output = model(src, tgt_input, SRC_PAD_IDX, TGT_PAD_IDX)
            output = output.view(-1, output.shape[-1])
            tgt_output = tgt_output.view(-1)
            loss = criterion(output, tgt_output)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        scheduler.step()

        epoch_loss = epoch_loss / len(train_loader)
        tune.report({"loss": epoch_loss, "epoch": epoch + 1})

# --- Main Sweep ---

if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)

    search_space = {
        "lr": tune.loguniform(5e-5, 2e-3),
        "weight_decay": tune.loguniform(1e-6, 1e-2),
        "batch_size": tune.choice([16, 32, 64]),
        "num_heads": tune.choice([4, 8]),
        "d_ff": tune.choice([1024, 2048]),
        "dropout": tune.uniform(0.1, 0.4),
        "epochs": 20,
        "d_model": 512,
        "num_layers": 6
    }

    optuna_search = OptunaSearch(metric="loss", mode="min")
    asha_scheduler = ASHAScheduler(
        time_attr='training_iteration',
        metric='loss',
        mode='min',
        max_t=20,
        grace_period=5,
        reduction_factor=2
    )

    tuner = tune.Tuner(
        tune.with_resources(train_tune, resources={"cpu": 1.0, "gpu": 1.0}),
        tune_config=tune.TuneConfig(
            search_alg=optuna_search,
            scheduler=asha_scheduler,
            num_samples=20,
            max_concurrent_trials=1, 
        ),
        param_space=search_space,
        run_config=ray.train.RunConfig() 
    )

    print("Starting Ray Tune hyperparameter sweep (20 trials)...")
    results = tuner.fit()
    print("Sweep completed!")

    best_result = results.get_best_result("loss", "min")
    print("Best hyperparameters found:", best_result.config)

    with open("/home/raid/vishal_kishore/assignment_04/tuning_results.pkl", "wb") as f:
        pickle.dump(best_result, f)
