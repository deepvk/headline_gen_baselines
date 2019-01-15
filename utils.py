from random import shuffle

import numpy as np
import pandas
import sentencepiece as spm
import ujson
from random import shuffle
from os.path import join
from nltk.tokenize import sent_tokenize


class DataLoader:
    def __init__(self, data_path=""):
        assert isinstance(
            data_path, str
        ), "Invalid data_path type. Required {}, but {} found".format(
            str, type(data_path)
        )
        self.data_path = data_path
        data = []
        with open(data_path + 'processed-ria.json', 'r') as file:
            for line in file:
                data += [ujson.loads(line)]

        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(join(data_path, "ria.model"))

        data_processed = []
        for val in data_processed:
            temp = [self.sp.EncodeAsIds(val['title'])]  # , self.sp.EncodeAsIds(val["body"])]
            sents = iter(sent_tokenize(val["body"]))
            sent = None
            while not sent:
                sent = next(sents)
                sent = sent.strip()
            temp.append(self.sp.EncodeAsIds(sent))

        indices = list(range(len(data)))
        shuffle(indices)

        data = [data[i] for i in indices]
        self.data = {'test': data[:20000], "valid": data[20000:30000], "train": data[30000:]}

    def save(self):
        self.save_headlines("train")
        self.save_headlines("valid")
        self.save_headlines("test")

        self.save_first_sents("train")
        self.save_first_sents("valid")
        self.save_first_sents("test")

    def save_headlines(self, part):
        with open(join(self.data_path, "{}_headlines.bpe".format(part)), "wt") as f:
            for val in self.data[part]:
                f.write(val[0] + "\n")

    def save_first_sents(self, part):
        with open(join(self.data_path, "{}_first_sents.bpe".format(part)), "wt") as f:
            for val in self.data[part]:
                f.write(val[1] + "\n")

    @staticmethod
    def load_json(path):
        with open(path, 'r') as f:
            return ujson.loads(f.read())