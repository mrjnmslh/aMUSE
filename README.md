## Unsupervised Alignment of Word Embeddings

This codebase is forked from facebook's MUSE which implements [Word Translation without Parallel Data](https://arxiv.org/pdf/1710.04087.pdf).

Additions to the forked codebase implement [Earth Mover’s Distance Minimization for Unsupervised Bilingual Lexicon Induction](http://aclweb.org/anthology/D17-1207).

## Dependencies
- Python 2/3 with numpy/scipy
- PyTorch

## Get evaluation datasets
Get monolingual and cross-lingual word embeddings evaluation datasets by simply running (in data/): `./get_evaluation.sh`

## Get monolingual word embeddings
For pre-trained monolingual word embeddings, download fastText Wikipedia embeddings.

You can download the English (en) and Chinese (zh) embeddings this way:

```
# English fastText Wikipedia embeddings
curl -Lo data/wiki.en.vec https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.en.vec
# Spanish fastText Wikipedia embeddings
curl -Lo data/wiki.zh.vec https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.zh.vec
```

## Training (CPU or GPU)
To train the Conneau et al. (2017) model run:

```python unsupervised.py --src_lang en --tgt_lang zh --src_emb data/wiki.en.vec --tgt_emb data/wiki.zh.vec```

To train the Conneau et al. (2017) model run:

```python unsupervised_wgan.py --src_lang en --tgt_lang zh --src_emb data/wiki.en.vec --tgt_emb data/wiki.zh.vec```

By default, the validation metric is the mean cosine of word pairs from a synthetic dictionary built with CSLS (Cross-domain similarity local scaling). Pay close attention to all default flags before running experiments.
