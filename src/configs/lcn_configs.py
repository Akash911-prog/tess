# Construction via add_pipe with custom config
transformer_configs = {
    "model": {
        "@architectures": "spacy-transformers.TransformerModel.v3",
        "name": "bert-base-uncased",
        "tokenizer_config": {"use_fast": True},
        "transformer_config": {"output_attentions": True},
        "mixed_precision": True,
        "grad_scaler_config": {"init_scale": 32768}
    }
}