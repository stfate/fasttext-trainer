from abc import ABC, abstractmethod
import copy
import re
import functools

import MeCab
import nltk

import lucia.tokenizer as tokenizer


ptn_number = re.compile(r"([0-9]|[０-９])+")


def get_tagger(dic_path):
    return MeCab.Tagger(f"-Ochasen -d {dic_path}")


def concat_continuous_numbers(tokens):
    prev_token = ""
    tokens_out = copy.deepcopy(tokens)
    for i in range( 1, len(tokens_out) )[::-1]:
        cur_token = tokens_out[i]
        prev_token = tokens_out[i-1]
        if ptn_number.match(prev_token) and ptn_number.match(cur_token):
            tokens_out[i-1] = prev_token + cur_token
            tokens_out.pop(i)
            i = i - 1

    return tokens_out


def tokenize(text, tagger):
    tokens = []
    parsed_text = tagger.parse(text)
    if parsed_text is not None:
        for line in parsed_text.split('\n'):
            if line == "EOS":
                break
            surface = line.split('\t')[0]
            tokens.append(surface)

        tokens = concat_continuous_numbers(tokens)
    
    return tokens


class DocumentTokenizerBase(ABC):
    @abstractmethod
    def tokenize(self, text):
        pass

    @abstractmethod
    def get_tokens_iterator(self, iter_docs):
        pass


class MecabDocumentTokenizer(DocumentTokenizerBase):
    def __init__(self, dic_path):
        self.tagger = tokenizer.MeCabTokenizer(dic_path)

    def tokenize(self, text, normalize=False):
        tokens,pos_tags = self.tagger.tokenize(text, normalize=normalize)
        tokens = concat_continuous_numbers(tokens)

        return tokens

    def get_tokens_iterator(self, iter_docs, normalize=False):
        tokenize = functools.partial(self.tokenize, normalize=normalize)

        def iter_tokens():
            for doc in iter_docs():
                yield tokenize(doc["body"])

        return iter_tokens


class NltkDocumentTokenizer(DocumentTokenizerBase):
    def __init__(self):
        self.tagger = tokenizer.NltkTokenizer()

    def tokenize(self, text, normalize=False):
        tokens,pos_tags = self.tagger.tokenize(text, normalize=normalize)
        tokens = concat_continuous_numbers(tokens)
        return tokens

    def get_tokens_iterator(self, iter_docs, normalize=False):
        tokenize = functools.partial(self.tokenize, normalize=normalize)

        def iter_tokens():
            for doc in iter_docs():
                yield tokenize(doc["body"])

        return iter_tokens
