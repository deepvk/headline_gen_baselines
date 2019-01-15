import ujson
from os.path import join
from random import shuffle

import sentencepiece as spm
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
        with open(join(data_path, 'processed-ria.json'), 'r') as file:
            for line in file:
                data += [ujson.loads(line)]

        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(join(data_path, "ria.model"))

        data_processed = []
        for val in data:
            temp = [" ".join(self.sp.EncodeAsPieces(val['title']))]  # , self.sp.EncodeAsIds(val["body"])]
            sents = iter(sent_tokenize(val["body"]))
            sent = None
            try:
                while not sent:
                    sent = next(sents)
                    sent = sent.strip()
            except StopIteration:
                sent = ""
            temp.append(" ".join(self.sp.EncodeAsPieces(sent)))
            data_processed.append(temp)

        indices = list(range(len(data)))
        shuffle(indices)

        data = [data_processed[i] for i in indices]
        self.data = {'test': data[:20000], "valid": data[20000:30000], "train": data[30000:]}
        del data_processed

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