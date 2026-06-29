
# Transformer From Scratch — English → French Translation

I built a Transformer encoder-decoder for English-to-French translation completely from scratch in PyTorch. No `nn.Transformer`, no pretrained weights — every piece (embeddings, positional encoding, multi-head attention, masking, the whole encoder-decoder stack) is hand-written. The goal wasn't to build the best possible translator, it was to actually understand how a Transformer works under the hood instead of treating it as a black box.

There's also a live Streamlit demo where you can type a sentence and see it translated, with greedy and beam search decoding and an option to peek at the model's cross-attention heatmap.

## Live Demo

[Try it here](https://transformer-translation-app-jux293.streamlit.app/)

## What's in here

I trained this on roughly 175K English-French sentence pairs, capped at 20 tokens per sentence so training didn't take forever. The model itself is 4 encoder layers + 4 decoder layers, `d_model=256`, 8 attention heads — nothing huge, but enough to learn real patterns.

| Metric | Value |
|---|---|
| Dataset | ~175K English-French sentence pairs |
| Max sentence length | 20 tokens |
| Architecture | 4 encoder + 4 decoder layers, `d_model=256`, 8 heads |
| Greedy BLEU (full test set) | *39* |
| Beam Search BLEU (full test set, beam=4) | **40** |

## Features

- A Transformer built layer by layer — embeddings, positional encoding, multi-head attention, masking, layer norm, feed-forward blocks, all from scratch
- Training with label smoothing and an LR warmup schedule
- Both greedy and beam search decoding, batched so evaluation doesn't take 40 minutes (it used to — more on that below)
- BLEU evaluation on a held-out test set
- A cross-attention heatmap so you can actually see what the model is "looking at" when it translates
- Honest qualitative error analysis — including the cases where it gets things wrong
- A live Streamlit app to try it yourself

## What I actually learned (and where it falls short)

I want to be upfront: this project is about understanding the architecture, not squeezing out the best possible BLEU score. A few real limitations worth knowing about:

- **The dataset is small** — ~175K pairs is nowhere close to what production translation systems train on
- **The sentences are simple** — mostly short, phrasebook-style text, not real-world, messy language
- **The vocabulary is fixed and word-level** — anything rare or unseen just becomes `<unk>`, and the translation breaks right there
- **It really cares about punctuation** — the model learned that punctuation usually signals "the sentence is ending," so if you leave it off, it can start repeating itself instead of stopping

### What the translations actually look like

Short, everyday sentences come out well, often matching the reference almost exactly. Where it struggles: rare words turning into `<unk>` and derailing an otherwise fine sentence, longer or more complex sentences losing their meaning along the way, and idioms getting translated word-for-word instead of idiomatically. Negation, at least, it handles correctly most of the time.

## How it's organized

```
├── app.py                  # Streamlit demo
├── model.py                # Transformer architecture
├── model_weights.pt        # Trained model weights
├── english_vocab.pkl       # English vocabulary
├── french_vocab.pkl        # French vocabulary
├── requirements.txt        # Dependencies
└── transformer-from-scratch.ipynb   # Full training notebook
```

## Running it yourself

```bash
pip install -r requirements.txt
streamlit run app.py
```

## The full notebook

If you want to see the whole process — data cleaning, the architecture, training, BLEU evaluation, attention visualization, error analysis — the full notebook is [here](https://www.kaggle.com/code/ramanpreet6728/transformer-from-scratch). 

## Built with

PyTorch · Streamlit · sacreBLEU
