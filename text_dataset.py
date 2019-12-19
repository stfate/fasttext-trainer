from abc import ABC, abstractmethod
import bz2
from pathlib import Path
import json
from glob import glob
import os
import re
import subprocess
import sys
import nltk

import lucia.textio as textio
import lucia.tokenizer as tokenizer


class TextDatasetBase(ABC):
    """
    a bass class for text dataset
    
    Attributes
    ----------
    """
    @abstractmethod
    def iter_docs(self):
        """
        iterator of documents
        
        Parameters
        ----------
        """
        yield None


class JapaneseText8Dataset(TextDatasetBase):
    """
    dataset for japanese text8
    
    Attributes
    ----------
    """
    def iter_docs(self, dataset_path):
        """
        get iterator of texts in one document
        
        Parameters
        ----------
        dataset_path: string
            path to dataset
        """
        with open(dataset_path, "r", encoding="utf-8") as fi:
            text = fi.read()
            sentence_tokenizer = tokenizer.JapaneseSentenceTokenizer()
            sentences = sentence_tokenizer.tokenize(text)
            for i,sentence in enumerate(sentences):
                sentence = sentence.replace(" ", "")
                yield {"title": i, "body": sentence}


class WikipediaDataset(TextDatasetBase):
    def __init__(self):
        self.__src_dir = os.path.dirname( os.path.abspath(__file__) )
        self.__wikiextractor_path = os.path.normpath("wikiextractor/WikiExtractor.py")
        self.__wikiextractor_cmd = os.path.join(self.__src_dir, self.__wikiextractor_path)
        self.__wikiextractor_cmd = os.path.normpath(self.__wikiextractor_cmd)

    def iter_docs(self, file_path, dir_path):
        """
        Parameters
        ----------
        file_path : string
            File path of wikipedia dump file
        dir_path : string
            Directory path where extracted text files are put
        """
        extracted_file_path_pattern = os.path.join(dir_path, "*", "*.bz2")
        extracted_file_paths = glob(extracted_file_path_pattern)
        if len(extracted_file_paths) == 0:
            cmd = ["python", self.__wikiextractor_cmd, "--compress", "--quiet", "-o", dir_path, file_path]
            subprocess.call(cmd, stdout=sys.stdout)
            extracted_file_paths = glob(extracted_file_path_pattern)

        re_doc_begin = re.compile(r"^<doc(\s.*)?>$")
        re_doc_end = re.compile(r"^</doc>$")
        re_nonstandard_namespace = re.compile(r".*:.*")
        for fpath in extracted_file_paths:
            with bz2.BZ2File(fpath) as f:
                is_inside_doc = False
                does_skip_this_doc = False
                lines = []
                for line in f:
                    line = line.decode("utf-8").rstrip()
                    if not is_inside_doc:
                        m = re_doc_begin.match(line)
                        if m is not None:
                            is_inside_doc = True
                            title = f.__next__().decode("utf-8").rstrip()
                            if re_nonstandard_namespace.match(title):
                                does_skip_this_doc = True
                            continue
                    else:
                        m = re_doc_end.match(line)
                        if m is not None:
                            if not does_skip_this_doc:
                                yield {"title": title, "body": "\n".join(lines)}
                            is_inside_doc = False
                            does_skip_this_doc = False
                            lines = []
                            continue
                        if len(line) > 0:
                            lines.append(line)


class ArtistReviewDataset(TextDatasetBase):
    def __init__(self):
        self.root_path = None

    def iter_docs(self, dataset_path):
        """
        get iterator of texts in one document
        
        Parameters
        ----------
        dataset_path: string
            path to dataset
        """
        self.root_path = Path(dataset_path)
        dataset_flist = self.root_path.glob("contents/*/*.html.txt")
        meta = json.load(open(self.root_path / "metadata.json", "r"))
        for fn in dataset_flist:
            fid = fn.parent.stem
            title = meta[fid]
            reader = textio.TextReader(fn)
            lines = reader.read()
            yield {"title": title, "body": lines}


class MARDDataset(TextDatasetBase):
    def __init__(self):
        self.root_path = None

    def iter_docs(self, dataset_path):
        """
        get iterator of texts in one document
        
        Parameters
        ----------
        dataset_path: string
            path to dataset
        """
        self.root_path = Path(dataset_path)
        reviews_json_fn = self.root_path / "mard_reviews.json"
        with open(reviews_json_fn, "r") as fi:
            for line in fi:
                review_dict = json.loads(line, encoding="utf-8")
                title = review_dict["reviewerID"]
                text = review_dict["reviewText"]
                yield {"title": title, "body": text}
