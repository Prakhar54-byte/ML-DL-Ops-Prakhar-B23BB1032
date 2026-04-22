import sacrebleu
from striprtf.striprtf import rtf_to_text

# ---- Read generated output ----
with open("output.txt", "r", encoding="utf-8") as f:
    predictions = [line.strip() for line in f.readlines() if line.strip()]

# ---- Read reference (RTF → plain text) ----
with open("output.rtf", "r", encoding="utf-8") as f:
    ref_text = rtf_to_text(f.read())

references = [line.strip() for line in ref_text.split("\n") if line.strip()]

# ---- Sanity check ----
min_len = min(len(predictions), len(references))
predictions = predictions[:min_len]
references = references[:min_len]

print(f"Evaluating {min_len} sentences...")

# ---- Compute BLEU ----
bleu = sacrebleu.corpus_bleu(predictions, [references])

print("BLEU score:", bleu.score)