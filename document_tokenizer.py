from abc import ABC, abstractmethod
import copy
import re
import functools
import MeCab

import lucia.tokenizer as tokenizer

ptn_number = re.compile(r"([0-9]|[０-９])+")


def concat_adjacent_numbers(tokens):
    prev_token = ""
    output_tokens = copy.deepcopy(tokens)
    for i in range( 1, len(output_tokens) )[::-1]:
        cur_token = output_tokens[i]
        prev_token = output_tokens[i-1]
        if ptn_number.match(prev_token) and ptn_number.match(cur_token):
            output_tokens[i-1] = prev_token + cur_token
            output_tokens.pop(i)
            i = i - 1

    return output_tokens


class DocumentTokenizerBase(ABC):
    """
    A bass class for document tokenizer
    
    Attributes
    ----------
    """
    @abstractmethod
    def tokenize(self, text):
        """
        tokenize text
        
        Parameters
        ----------
        text: string
            input text
        
        Returns
        -------
        tokens: list of string
            tokenized text sequences
        """
        pass


class MecabDocumentTokenizer(DocumentTokenizerBase):
    def __init__(self, dic_path):
        """        
        Parameters
        ----------
        dic_path: string
            path to dictionary

        Returns
        -------
        """
        self.tagger = tokenizer.MeCabWordTokenizer(dic_path)

    def tokenize(self, text, normalize=False):
        """
        tokenize text
        
        Parameters
        ----------
        text: string
            input text
        normalize: bool
            a switch for normalization
        
        Returns
        -------
        tokens: list of string
            tokenized text
        """
        tokens,pos_tags = self.tagger.tokenize(text, normalize=normalize)
        tokens = concat_adjacent_numbers(tokens)

        return tokens


class NltkDocumentTokenizer(DocumentTokenizerBase):
    def __init__(self):
        self.tagger = tokenizer.NltkWordTokenizer()

    def tokenize(self, text, normalize=False):
        """
        tokenize text
        
        Parameters
        ----------
        text: string
            input text
        normalize: bool
            a switch for normalization
        
        Returns
        -------
        tokens: list of string
            tokenized text
        """
        tokens,pos_tags = self.tagger.tokenize(text, normalize=normalize)
        tokens = concat_adjacent_numbers(tokens)
        return tokens
