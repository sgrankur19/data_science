"""
Optimized training script for SQuAD v1.1 QA task.
Quick wins implemented:
- Limit encoder/decoder vocab sizes (`num_words`) to reduce softmax and embedding sizes
- Smaller `EMBEDDING_DIM` and `LATENT_DIM`
- Use `tf.data` pipeline with batching and prefetch
- Optional mixed precision when a GPU is available
- Configurable `MAX_SAMPLES` and `EPOCHS` for fast testing

Run: python optimized_train.py --max_samples 5000 --epochs 2
"""
import json
import re
import argparse
from pathlib import Path
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import (
    Input, Embedding, Bidirectional, LSTM, Dense, Concatenate,
    AdditiveAttention, TimeDistributed
)
from tensorflow.keras.models import Model

def train(
    data_path: str = "train-v1.1.json",
    max_samples: int = 2000,
    batch_size: int = 128,
    epochs: int = 2,
    encoder_vocab: int = 20000,
    decoder_vocab: int = 8000,
    embedding_dim: int = 64,
    latent_dim: int = 64,
    mixed_precision: bool = False,
    save_dir: str = "output",
):
    """Train the optimized model and return useful artifacts.

    Returns a dict with keys: model, encoder_tokenizer, decoder_tokenizer,
    context_pad, question_pad, decoder_input, decoder_target, context_len,
    question_len, decoder_len, encoder_vocab_size, decoder_vocab_size
    """
    DATA_PATH = Path(data_path)
    if mixed_precision:
        try:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
        except Exception:
            pass

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"{DATA_PATH} not found. Download SQuAD v1.1 and place it here.")

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        squad = json.load(f)

    records = []
    for article in squad["data"]:
        for paragraph in article.get("paragraphs", []):
            context = paragraph.get("context", "")
            for qa in paragraph.get("qas", []):
                answers = qa.get("answers", [])
                if answers:
                    records.append((context, qa.get("question", ""), answers[0].get("text", ""), answers[0].get("answer_start", -1)))

    records = records[:max_samples]
    contexts, questions, answers, _ = zip(*records)

    def _clean(text):
        t = str(text).lower()
        t = re.sub(r"[^a-z0-9.,!?';:\-\s]", " ", t)
        return re.sub(r"\s+", " ", t).strip()

    contexts = [_clean(t) for t in contexts]
    questions = [_clean(t) for t in questions]
    answers = [_clean(t) for t in answers]

    encoder_tokenizer = Tokenizer(oov_token="<oov>", num_words=encoder_vocab)
    encoder_tokenizer.fit_on_texts(list(contexts) + list(questions))

    decoder_tokenizer = Tokenizer(oov_token="<oov>", num_words=decoder_vocab, filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n')
    answers_with_markers = ["<start> " + a + " <end>" for a in answers]
    decoder_tokenizer.fit_on_texts(answers_with_markers)

    context_seq = encoder_tokenizer.texts_to_sequences(contexts)
    question_seq = encoder_tokenizer.texts_to_sequences(questions)
    decoder_seq = decoder_tokenizer.texts_to_sequences(answers_with_markers)

    context_len = int(np.percentile([len(s) for s in context_seq], 95))
    question_len = int(np.percentile([len(s) for s in question_seq], 95))
    decoder_len = max(10, int(np.percentile([len(s) for s in decoder_seq], 95)))

    context_pad = pad_sequences(context_seq, maxlen=context_len, padding="post")
    question_pad = pad_sequences(question_seq, maxlen=question_len, padding="post")
    decoder_pad = pad_sequences(decoder_seq, maxlen=decoder_len, padding="post")

    decoder_input = decoder_pad[:, :-1]
    decoder_target = decoder_pad[:, 1:]

    encoder_vocab_size = min(encoder_vocab, len(encoder_tokenizer.word_index) + 1)
    decoder_vocab_size = min(decoder_vocab, len(decoder_tokenizer.word_index) + 1)

    dataset = tf.data.Dataset.from_tensor_slices((
        {
            "context_input": context_pad.astype("int32"),
            "question_input": question_pad.astype("int32"),
            "decoder_input": decoder_input.astype("int32"),
        },
        decoder_target.astype("int32"),
    ))
    dataset = dataset.shuffle(1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Build model
    context_input = Input(shape=(context_len,), name="context_input")
    question_input = Input(shape=(question_len,), name="question_input")
    decoder_input_layer = Input(shape=(decoder_len - 1,), name="decoder_input")

    encoder_embedding = Embedding(encoder_vocab_size, embedding_dim, mask_zero=True, name="encoder_embedding")
    context_emb = encoder_embedding(context_input)
    question_emb = encoder_embedding(question_input)

    context_encoded = Bidirectional(LSTM(latent_dim, return_sequences=True), name="context_bilstm")(context_emb)
    question_encoded = Bidirectional(LSTM(latent_dim, return_sequences=True), name="question_bilstm")(question_emb)

    from tensorflow.keras.layers import GlobalAveragePooling1D, RepeatVector
    question_pool = GlobalAveragePooling1D(name="question_pool")(question_encoded)
    question_rep = RepeatVector(context_len, name="question_repeat")(question_pool)
    combined_encoder = Concatenate(name="context_question_stack")([context_encoded, question_rep])

    decoder_embedding = Embedding(decoder_vocab_size, embedding_dim, mask_zero=True, name="decoder_embedding")
    dec_emb = decoder_embedding(decoder_input_layer)

    decoder_lstm = LSTM(2 * latent_dim, return_sequences=True, name="decoder_lstm")
    dec_outputs = decoder_lstm(dec_emb)

    from tensorflow.keras.layers import TimeDistributed, Dense as KerasDense
    encoder_proj = TimeDistributed(KerasDense(2 * latent_dim), name="encoder_projection")(combined_encoder)
    attention = AdditiveAttention(name="attention")([dec_outputs, encoder_proj])

    combined = Concatenate(name="decoder_concat")([dec_outputs, attention])
    logits = TimeDistributed(Dense(decoder_vocab_size, activation="softmax"), name="token_classifier")(combined)

    model = Model([context_input, question_input, decoder_input_layer], logits, name="optimized_seq2seq")
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

    model.summary()

    start = time.time()
    history = model.fit(dataset, epochs=epochs)
    end = time.time()
    print(f"Training finished in {end - start:.1f}s")

    out = Path(save_dir)
    out.mkdir(exist_ok=True)
    import pickle
    with open(out / "encoder_tokenizer.pkl", "wb") as f:
        pickle.dump(encoder_tokenizer, f)
    with open(out / "decoder_tokenizer.pkl", "wb") as f:
        pickle.dump(decoder_tokenizer, f)

    model_save_path = out / "optimized_model.keras"
    model.save(model_save_path)
    print(f"Saved tokenizers and model to {out}/")

    return {
        "model": model,
        "encoder_tokenizer": encoder_tokenizer,
        "decoder_tokenizer": decoder_tokenizer,
        "context_pad": context_pad,
        "question_pad": question_pad,
        "decoder_input": decoder_input,
        "decoder_target": decoder_target,
        "context_len": context_len,
        "question_len": question_len,
        "decoder_len": decoder_len,
        "encoder_vocab_size": encoder_vocab_size,
        "decoder_vocab_size": decoder_vocab_size,
        "history": history,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="train-v1.1.json")
    parser.add_argument("--max_samples", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--encoder_vocab", type=int, default=20000)
    parser.add_argument("--decoder_vocab", type=int, default=8000)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--save_dir", default="output")
    args = parser.parse_args()

    train(
        data_path=args.data,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        epochs=args.epochs,
        encoder_vocab=args.encoder_vocab,
        decoder_vocab=args.decoder_vocab,
        embedding_dim=args.embedding_dim,
        latent_dim=args.latent_dim,
        mixed_precision=args.mixed_precision,
        save_dir=args.save_dir,
    )
