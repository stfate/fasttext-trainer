import functools
from pathlib import Path
import multiprocessing
import logging
import more_itertools
from gensim.models.fasttext import FastText

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s:%(message)s", datefmt="%Y/%m/%d %H:%M:%S")


def train_fasttext_model(output_model_path, iter_docs, tokenizer, size=300, window=8, min_count=5, sg=1, epoch=5):
    """
    Parameters
    ----------
    output_model_path : string
        path of fastText model
    iter_docs : iterator
        iterator of documents, which are raw texts
    tokenizer : subclass of DocumentTokenizerBase
        word tokenizer
    size : int
        size of word vector
    window : int
        window size of word2vec
    min_count : int
        minimum word count
    sg : int
        word2vec training algorithm (1: skip-gram other:CBOW)
    epoch : int
        number of epochs
    """
    logging.info("get tokens iterator")

    iter_tokens = tokenizer.get_tokens_iterator(iter_docs, normalize=False)

    logging.info("count number of documents")

    n_obs = more_itertools.ilen(iter_tokens())

    logging.info("build vocabulary")

    model = FastText(
        size=size,
        window=window,
        min_count=min_count,
        sg=sg,
        workers=multiprocessing.cpu_count()
    )
    model.build_vocab(iter_tokens())
    
    logging.info("train fasttext")

    model.train(iter_tokens(), total_examples=n_obs, epochs=epoch)
    model.init_sims(replace=True)

    logging.info("save model")

    p = Path(output_model_path)
    if not p.parent.exists():
        p.parent.mkdir(parents=True)
    model.save(output_model_path)

    logging.info("done.")
