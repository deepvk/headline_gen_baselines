# Baselines for Headline Generation on "Rossiya Segodnya" dataset

Data preparation for first sentence and OpenNMT baselines. You need to place `processed-ria.json` into `data` folder. 
We have excluded HTML tags from "Rossiya Segodnya" corpus to obtain `processed-ria.json`.
Run `python main.py` to get data prepared, but you need to download corpus by yourself.

To test first sentence baseline you need to run 
```bash
python get_rouge.py data/test_first_sents.bpe data/test_headlines.bpe
```

To train OpenNMT baseline, just follow instructions from official [OpenNMT repository](https://github.com/OpenNMT/OpenNMT-py):
```bash
python preprocess.py -train_src data/train_first_sents.bpe -train_tgt data/train_headlines.bpe -valid_src data/valid_first_sents.bpe -valid_tgt data/valid_headlines.bpe -save_data data/ria
python train.py -data data/ria -save_model ria-model
```

To test OpenNMT:
```bash
python translate.py -model ria-model_XXX.pt -src data/test_first_sents.bpe -output pred.bpe -replace_unk
python get_rouge.py pred.bpe data/test_headlines.bpe
```
