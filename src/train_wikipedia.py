import argparse
from functools import partial
import tempfile

import document_tokenizer
import dictionary_downloader
import text_corpus
import fasttext_trainer

__jawiki_dump_file_name = "jawiki-latest-pages-articles.xml.bz2"
__jawiki_dump_url = f"https://dumps.wikimedia.org/jawiki/latest/{__jawiki_dump_file_name}" # noqa


def get_options():
    parser = argparse.ArgumentParser()

    parser.add_argument("--build-model", action="store_true", default=False)
    parser.add_argument("-o", "--output-model-path", default="model/word2vec.gensim.model")
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--min-count", type=int, default=10)

    parser.add_argument("--download-wikipedia-dump", action="store_true", default=False)
    parser.add_argument("--wikipedia-dump-path", default=f"data/{__jawiki_dump_file_name}")
    parser.add_argument("--wikipedia-dump-url", default=__jawiki_dump_url)
    parser.add_argument("--lang", default="ja")

    parser.add_argument("--download-neologd", action="store_true", default=False)
    parser.add_argument("--dictionary-path", default="output/dic")

    parser.add_argument("--use-pretrained-model", action="store_true", default=False)
    parser.add_argument("--pretrained-model-path", default="model/pretrained.word2vec.gensim.model")

    args = parser.parse_args()
    return vars(args)

if __name__ == "__main__":
    options = get_options()

    commands = [
        "download_neologd",
        "download_wikipedia_dump",
        "build_model"
    ]
    if not any(options[c] for c in commands):
        print("At least one of following options needs to be specified:",
              *["  --" + c.replace("_", "-") for c in commands], sep="\n")

    dic_path = options["dictionary_path"]
    if options["download_neologd"]:
        dictionary_downloader.download_neologd(dic_path)

    wikipedia_dump_path = options["wikipedia_dump_path"]
    wikipedia_dump_url = options["wikipedia_dump_url"]
    wikipedia = text_corpus.WikipediaCorpus()
    if options["download_wikipedia_dump"]:
        wikipedia.download_dump(wikipedia_dump_path, wikipedia_dump_url)

    output_model_path = options["output_model_path"]
    size = options["size"]
    window = options["window"]
    min_count = options["min_count"]
    use_pretrained_model = options["use_pretrained_model"]
    pretrained_model_path = options["pretrained_model_path"]
    lang = options["lang"]
    if options["build_model"]:
        with tempfile.TemporaryDirectory() as temp_dir:
            iter_docs = partial(wikipedia.iter_docs, wikipedia_dump_path, temp_dir)
            if lang == "ja":
                tokenizer = document_tokenizer.MecabDocumentTokenizer(dic_path)
            elif lang == "en":
                tokenizer = document_tokenizer.NltkDocumentTokenizer()
            fasttext_trainer.train_fasttext_model(output_model_path, iter_docs, tokenizer, size, window, min_count, use_pretrained_model, pretrained_model_path)
