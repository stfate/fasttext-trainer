import argparse
from functools import partial

import document_tokenizer
import text_dataset
import fasttext_trainer


def get_options():
    parser = argparse.ArgumentParser()

    parser.add_argument("-o", "--output-model-path", default="model/word2vec.gensim.model")
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--min-count", type=int, default=10)
    parser.add_argument("--sg", type=int, default=1)
    parser.add_argument("--epoch", type=int, default=5)

    parser.add_argument("--dataset-path", default="data/TextDataset")
    parser.add_argument("--lang", default="ja")

    parser.add_argument("--dictionary-path", default="output/dic")

    parser.add_argument("--use-pretrained-model", action="store_true", default=False)
    parser.add_argument("--pretrained-model-path", default="model/pretrained.word2vec.gensim.model")

    args = parser.parse_args()
    return vars(args)


if __name__ == "__main__":
    options = get_options()

    output_model_path = options["output_model_path"]
    size = options["size"]
    window = options["window"]
    min_count = options["min_count"]
    sg = options["sg"]
    epoch = options["epoch"]
    dataset_path = options["dataset_path"]
    dic_path = options["dictionary_path"]
    use_pretrained_model = options["use_pretrained_model"]
    pretrained_model_path = options["pretrained_model_path"]
    lang = options["lang"]
    # dataset = text_dataset.ArtistReviewDataset()
    dataset = text_dataset.MARDDataset()
    
    iter_docs = partial(dataset.iter_docs, dataset_path)
    if lang == "ja":
        tokenizer = document_tokenizer.MecabDocumentTokenizer(dic_path)
    elif lang == "en":
        tokenizer = document_tokenizer.NltkDocumentTokenizer()
    fasttext_trainer.train_fasttext_model(output_model_path, iter_docs, tokenizer, size, window, min_count, sg, epoch, use_pretrained_model, pretrained_model_path)
