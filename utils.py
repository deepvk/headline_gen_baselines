import ujson
from os.path import join
from random import shuffle

import sentencepiece as spm
from nltk.tokenize import sent_tokenize


class DataLoader:
    def __init__(self, data_path="", seed=42, name="ria"):
        assert isinstance(
            data_path, str
        ), "Invalid data_path type. Required {}, but {} found".format(
            str, type(data_path)
        )
        self.data_path = data_path
        self.seed = seed
        data = []
        with open(join(data_path, 'processed-{}.json'.format(name)), 'r') as file:
            for line in file:
                data += [ujson.loads(line)]

        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(join(data_path, "{}.model".format(name)))

        data_processed = []
        for val in data:
            if name == "ria":
                title = val['title']
                body = val["body"]
            else:
                title = val[0]
                body = val[1]
            temp = [" ".join(self.sp.EncodeAsPieces(title))]  # , self.sp.EncodeAsIds(val["body"])]
            sents = sent_tokenize(body)
            if len(sents):
                sent = sents[0]
                if len(sents) > 1 and name == "ria":
                    sent = sents[1]
            else:
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
        with open(join(self.data_path, "{}_headlines_{}.bpe".format(part, self.seed)), "wt") as f:
            for val in self.data[part]:
                f.write(val[0] + "\n")

    def save_first_sents(self, part):
        with open(join(self.data_path, "{}_first_sents_{}.bpe".format(part, self.seed)), "wt") as f:
            for val in self.data[part]:
                f.write(val[1] + "\n")

    @staticmethod
    def load_json(path):
        with open(path, 'r') as f:
            return ujson.loads(f.read())


def unbpe(text):
    tokens = text.split()
    words = []
    for token in tokens:
        if not token.startswith("▁"):
            words[-1] += token
        else:
            words.append(token)
    if not words:
        words = ["▁"]
    return " ".join(words)
