import re
import pickle

import streamlit as st
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from model import build_model, create_padding_mask, generate_look_ahead_mask

MAX_LEN = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------- preprocessing ----------

def ensure_punctuation(text):
    text = text.strip()
    if text and text[-1] not in ".!?":
        text += "."
    return text

def preprocess(text):
    text = text.lower()
    text = re.sub(r'([.!?,;:])', r' \1 ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def detokenize(text):
    text = re.sub(r'\s+([.!?,;:])', r'\1', text)
    return text.strip()


# ---------- cached loading ----------

@st.cache_resource
def load_assets():
    with open("english_vocab.pkl", "rb") as f:
        english_vocab = pickle.load(f)
    with open("french_vocab.pkl", "rb") as f:
        french_vocab = pickle.load(f)

    idx_to_french = {idx: word for word, idx in french_vocab.items()}

    model = build_model(
        src_vocab_size=len(english_vocab),
        trg_vocab_size=len(french_vocab),
        src_max_len=MAX_LEN,
        trg_max_len=MAX_LEN + 2,
        device=DEVICE
    )
    model.load_state_dict(torch.load("model_weights.pt", map_location=DEVICE))
    model.eval()

    return model, english_vocab, french_vocab, idx_to_french


def english_word_to_idx(sentence, english_vocab):
    return [english_vocab.get(word, english_vocab["<unk>"]) for word in sentence.split()]


def ids_to_text(ids, idx_to_french):
    return " ".join(idx_to_french.get(i, "<unk>") for i in ids)


# ---------- greedy decode ----------

def greedy_decode(model, src_ids, pad_idx, sos_idx, eos_idx, max_len=20, device="cpu"):
    model.eval()
    src = torch.tensor(src_ids).unsqueeze(0).to(device)

    with torch.no_grad():
        src_padding_mask = create_padding_mask(src, pad_idx)
        encoder_output = model.encoder(src, src_padding_mask)

        decoded = torch.tensor([[sos_idx]], device=device)

        for _ in range(max_len):
            look_ahead_mask = generate_look_ahead_mask(decoded.size(1), device)
            decoder_output = model.decoder(decoded, encoder_output, look_ahead_mask, src_padding_mask)
            logits = model.fc_out(decoder_output[:, -1, :])
            next_token = logits.argmax(dim=-1).unsqueeze(0)
            decoded = torch.cat([decoded, next_token], dim=1)
            if next_token.item() == eos_idx:
                break

    out_ids = decoded.squeeze(0).tolist()[1:]
    if eos_idx in out_ids:
        out_ids = out_ids[:out_ids.index(eos_idx)]
    return out_ids


# ---------- beam search decode ----------

def beam_search_decode(model, src_ids, pad_idx, sos_idx, eos_idx, max_len=20, beam_size=4, length_penalty=1.0, device="cpu"):
    model.eval()
    src = torch.tensor(src_ids).unsqueeze(0).to(device)

    with torch.no_grad():
        src_padding_mask = create_padding_mask(src, pad_idx)
        encoder_output = model.encoder(src, src_padding_mask)

        beams = [([sos_idx], 0.0)]
        completed = []

        for _ in range(max_len):
            new_beams = []
            for seq, score in beams:
                if seq[-1] == eos_idx:
                    completed.append((seq, score))
                    continue

                trg_input = torch.tensor(seq).unsqueeze(0).to(device)
                look_ahead_mask = generate_look_ahead_mask(trg_input.size(1), device)

                decoder_output = model.decoder(trg_input, encoder_output, look_ahead_mask, src_padding_mask)
                logits = model.fc_out(decoder_output[:, -1, :])
                log_probs = F.log_softmax(logits, dim=-1).squeeze(0)

                topk_log_probs, topk_ids = log_probs.topk(beam_size)
                for lp, idx in zip(topk_log_probs.tolist(), topk_ids.tolist()):
                    new_beams.append((seq + [idx], score + lp))

            if not new_beams:
                break

            new_beams.sort(key=lambda x: x[1] / (len(x[0]) ** length_penalty), reverse=True)
            beams = new_beams[:beam_size]

            if all(seq[-1] == eos_idx for seq, _ in beams):
                completed.extend(beams)
                break

        completed.extend(beams)
        completed.sort(key=lambda x: x[1] / (len(x[0]) ** length_penalty), reverse=True)
        best_seq = completed[0][0]

    best_seq = best_seq[1:]
    if eos_idx in best_seq:
        best_seq = best_seq[:best_seq.index(eos_idx)]
    return best_seq


# ---------- attention extraction ----------

def get_cross_attention(model, sentence, out_ids, english_vocab, french_vocab, device):
    pad_idx = english_vocab["<pad>"]
    sos_idx = french_vocab["<sos>"]
    
    sentence = ensure_punctuation(sentence)
    text = preprocess(sentence)
    src_ids = english_word_to_idx(text, english_vocab)
    src_tensor = torch.tensor(src_ids).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        src_padding_mask = create_padding_mask(src_tensor, pad_idx)
        encoder_output = model.encoder(src_tensor, src_padding_mask)

        trg_input = torch.tensor([sos_idx] + out_ids).unsqueeze(0).to(device)
        look_ahead_mask = generate_look_ahead_mask(trg_input.size(1), device)
        _ = model.decoder(trg_input, encoder_output, look_ahead_mask, src_padding_mask)

    attn_weights = model.decoder.layers[-1].last_attn_weights  # [1, num_heads, trg_len, src_len]
    attn_avg = attn_weights[0].mean(dim=0).cpu()                # average across heads -> [trg_len, src_len]

    src_tokens = text.split()
    return attn_avg, src_tokens


def plot_attention_heatmap(attn_avg, src_tokens, trg_tokens):
    fig, ax = plt.subplots(figsize=(max(4, len(src_tokens) * 0.8), max(3, len(trg_tokens) * 0.8)))
    im = ax.imshow(attn_avg.numpy(), cmap="viridis")
    ax.set_xticks(range(len(src_tokens)))
    ax.set_xticklabels(src_tokens, rotation=45, ha="right")
    ax.set_yticks(range(len(trg_tokens)))
    ax.set_yticklabels(trg_tokens)
    ax.set_xlabel("Source (English)")
    ax.set_ylabel("Generated (French)")
    fig.colorbar(im, ax=ax, label="Attention weight")
    fig.tight_layout()
    return fig


def translate(sentence, model, english_vocab, french_vocab, idx_to_french, use_beam=True, beam_size=4):
    pad_idx = english_vocab["<pad>"]
    sos_idx = french_vocab["<sos>"]
    eos_idx = french_vocab["<eos>"]
    
    sentence = ensure_punctuation(sentence)
    text = preprocess(sentence)
    ids = english_word_to_idx(text, english_vocab)

    if use_beam:
        out_ids = beam_search_decode(model, ids, pad_idx, sos_idx, eos_idx, max_len=MAX_LEN, beam_size=beam_size, device=DEVICE)
    else:
        out_ids = greedy_decode(model, ids, pad_idx, sos_idx, eos_idx, max_len=MAX_LEN, device=DEVICE)

    translation = detokenize(ids_to_text(out_ids, idx_to_french))
    return translation, out_ids

st.set_page_config(page_title="English to French Translator", page_icon="🇫🇷")

st.title("English → French Translator")
st.caption("A Transformer model built from scratch in PyTorch.")

with st.expander("⚠️ About this model's limitations"):
    st.markdown("""
This project focuses on building the Transformer architecture from scratch, not on improving translation accuracy.

Prediction quality is limited by:

- **Small dataset**: about 175K sentence pairs, much less than what production translation systems use.

- **Simple, short sentences**: the training data mainly consists of short, phrasebook-style sentences, not general-domain text.

- **Fixed, word-level vocabulary**: rare or unseen words are replaced with `<unk>` and cannot be translated.

- **Sensitive to missing punctuation**: the model learned to link punctuation with sentence endings; inputs lacking punctuation may produce repetitive or incomplete output.

View this as a demonstration of the Transformer architecture and training mechanics, not a production-quality translator.
""")

model, english_vocab, french_vocab, idx_to_french = load_assets()

decoding_choice = st.radio(
    "Decoding strategy:",
    options=["Beam Search (better quality, slower)", "Greedy (faster)"],
    index=0
)
use_beam = decoding_choice.startswith("Beam")

user_input = st.text_input("Enter an English sentence:", placeholder="e.g. I love you.")
show_attention = st.checkbox("Show cross-attention heatmap", value=False)

if "translation" not in st.session_state:
    st.session_state.translation = None
    st.session_state.out_ids = None

if st.button("Translate"):
    if user_input.strip():
        st.session_state.translation, st.session_state.out_ids = translate(
            user_input, model, english_vocab, french_vocab, idx_to_french, use_beam=use_beam
        )
    else:
        st.warning("Please enter a sentence first.")

if st.session_state.translation:
    st.markdown("### Translation")
    st.success(st.session_state.translation)

    if show_attention and len(st.session_state.out_ids) > 0:
        attn_avg, src_tokens = get_cross_attention(model, user_input, st.session_state.out_ids, english_vocab, french_vocab, DEVICE)
        trg_tokens = [ids_to_text([t], idx_to_french) for t in st.session_state.out_ids]
        st.markdown("### Cross-Attention Heatmap")
        st.caption("Shows which English words the model focused on when generating each French word.")
        fig = plot_attention_heatmap(attn_avg, src_tokens, trg_tokens)
        st.pyplot(fig)

st.divider()
