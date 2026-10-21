# FinSage — Financial News Sentiment Classification Using Deep Learning

FinSage analyzes financial news headlines and classifies their sentiment as **positive**, **negative**, or **neutral**. Built with Python and TensorFlow/Keras, it explores Recurrent Neural Networks (RNNs), Long Short-Term Memory (LSTM) networks, and Bidirectional LSTMs to understand contextual patterns in financial text.

Using the **Financial PhraseBank** dataset, FinSage incorporates text preprocessing, a traditional machine-learning baseline, systematic model evaluation, and controlled ablation studies to investigate the impact of architectural choices and hyperparameters on classification performance. The project also examines class imbalance, selection bias, data leakage, and model generalization to promote reliable and reproducible results.

## Features

- **Traditional baseline**: TF-IDF + Logistic Regression / Linear SVM
- **Deep learning models**: Simple RNN, LSTM, Bidirectional LSTM
- **No data leakage**: tokenizer is fit only on the training split
- **Class imbalance handling**: balanced/auto class weights, optional focal loss
- **Reproducibility**: global seeding, deterministic ops, config-driven experiments
- **Experiment tracking**: MLflow (params, metrics, artifacts, model registry)
- **Ablation studies**: embedding dim, hidden units, layers, dropout, bidirectionality, LR, batch size, sequence length, optimizer
- **Evaluation**: macro F1, per-class precision/recall/F1, confusion matrices, ROC-AUC

## Project Structure

```
FinSage/
├── configs/                # Hydra configs (data, model, train)
├── src/finsage/
│   ├── data/               # Loader, preprocessing, stratified splits (no leakage)
│   ├── models/             # Baseline, RNN, LSTM, BiLSTM
│   ├── training/           # Trainer + callbacks (EarlyStopping, MLflow, ...)
│   ├── evaluation/         # Metrics + visualization
│   ├── experiments/        # MLflow tracking + ablation framework
│   └── utils/              # Reproducibility seeding
├── tests/                  # Unit tests
└── pyproject.toml          # Dependencies (uv)
```

## Dataset

[Financial PhraseBank](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news) — sentences labeled `negative`, `neutral`, or `positive`.

The dataset is auto-downloaded on first run from the [ahmedrachid/FinancialPhraseBank](https://github.com/ahmedrachid/FinancialPhraseBank) mirror into `data/raw/`.

| Variant | Samples | Default |
|---------|---------|---------|
| `50Agree` | 4,840 | ✅ |
| `75Agree` | ~3,453 | |
| `100Agree` | ~2,264 | |

## Installation

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Usage

```bash
# Train a single model (defaults to LSTM)
uv run python -m finsage

# Override config via Hydra
uv run python -m finsage model.type=bilstm train.epochs=50

# Compare architectures
uv run python -m finsage mode=compare model_types='["rnn","lstm","bilstm"]'

# Run an ablation study
uv run python -m finsage mode=ablation ablation_grid=embedding_dim

# View MLflow UI
mlflow ui --port 5000
```

### Available ablation grids

`embedding_dim`, `hidden_units`, `num_layers`, `dropout`, `bidirectional`, `learning_rate`, `batch_size`, `seq_length`, `optimizer`

## Testing

```bash
uv run python -m pytest tests/ -v
```

## Configuration

All experiments are driven by Hydra configs in `configs/`:

- `data.yaml` — dataset variant, split ratios, preprocessing, vocabulary, batching
- `model.yaml` — architecture type, embedding, RNN/LSTM layers, baseline settings
- `train.yaml` — epochs, optimizer, loss, class weights, callbacks, MLflow, seeding

## Reproducibility

Splits, tokenizer fitting, and model initialization are all seeded (`train.seed`). Deterministic TensorFlow ops are enabled via `train.deterministic`. Tokenizers are fit strictly on the training split to prevent data leakage.

## License

Released under the [MIT License](LICENSE).
