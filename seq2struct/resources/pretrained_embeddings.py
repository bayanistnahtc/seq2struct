import abc
import functools
import os
import pickle
import time

import bpemb
import corenlp
import nltk
import torch
import torchtext
from pytorch_pretrained_bert import BertTokenizer, BertModel

from seq2struct.resources import corenlp
from seq2struct.utils import registry


class Embedder(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def tokenize(self, sentence):
        '''Given a string, return a list of tokens suitable for lookup.'''
        pass

    @abc.abstractmethod
    def untokenize(self, tokens):
        '''Undo tokenize.'''
        pass

    @abc.abstractmethod
    def lookup(self, token):
        '''Given a token, return a vector embedding if token is in vocabulary.

        If token is not in the vocabulary, then return None.'''
        pass

    @abc.abstractmethod
    def contains(self, token):
        pass

    @abc.abstractmethod
    def to(self, device):
        '''Transfer the pretrained embeddings to the given device.'''
        pass


@registry.register('word_emb', 'glove')
class GloVe(Embedder):

    def __init__(self, kind):
        cache = os.path.join(os.environ.get('CACHE_DIR', os.getcwd()), '.vector_cache')
        self.glove = torchtext.vocab.GloVe(name=kind, cache=cache)
        self.dim = self.glove.dim
        self.vectors = self.glove.vectors

    @functools.lru_cache(maxsize=1024)
    def tokenize(self, text):
        ann = corenlp.annotate(text, annotators=['tokenize', 'ssplit'])
        return [tok.word.lower() for sent in ann.sentence for tok in sent.token]

    def untokenize(self, tokens):
        return ' '.join(tokens)

    def lookup(self, token):
        i = self.glove.stoi.get(token)
        if i is None:
            return None
        return self.vectors[i]

    def contains(self, token):
        return token in self.glove.stoi

    def to(self, device):
        self.vectors = self.vectors.to(device)


@registry.register('word_emb', 'bpemb')
class BPEmb(Embedder):
    def __init__(self, dim, vocab_size, lang='en'):
        self.bpemb = bpemb.BPEmb(lang=lang, dim=dim, vs=vocab_size)
        self.dim = dim
        self.vectors = torch.from_numpy(self.bpemb.vectors)

    def tokenize(self, text):
        return self.bpemb.encode(text)

    def untokenize(self, tokens):
        return self.bpemb.decode(tokens)

    def lookup(self, token):
        i = self.bpemb.spm.PieceToId(token)
        if i == self.bpemb.spm.unk_id():
            return None
        return self.vectors[i]

    def contains(self, token):
        return self.lookup(token) is not None

    def to(self, device):
        self.vectors = self.vectors.to(device)


@registry.register('word_emb', 'tfidfemb')
class TFIDF(Embedder):

    def __init__(self, kind):
        path = os.path.join(os.environ.get('CACHE_DIR', os.getcwd()), 'seq2struct/resources/tfidf.pkl')
        self.tfidf = pickle.load(open(path, "rb")) #"~/Documents/synthesis/seq2struct_tfidf/seq2struct/resources/tfidf.pkl"
        self.dim = len(self.tfidf.vocabulary_)

    @functools.lru_cache(maxsize=1024)
    def tokenize(self, text):
        text = text.replace('[;\", \.]', ' ').lower()
        return nltk.word_tokenize(text)

    def untokenize(self, tokens):
        return ' '.join(tokens)

    def lookup(self, token):
        tfidf_vec = self.tfidf.transform([token])[0]
        return torch.from_numpy(tfidf_vec.toarray()[0])


    def contains(self, token):
        return token in self.tfidf.vocabulary_

    def to(self, device):
        # self.vectors = self.vectors.to(device)
        pass


@registry.register('word_emb', 'bertemb')
class BertEmb(Embedder):
    # # Load pre-trained model (weights)
    # model = BertModel.from_pretrained('bert-base-uncased')
    # model.eval()
    #
    # # If you have a GPU, put everything on cuda
    # tokens_tensor = tokens_tensor.to('cuda')
    # segments_tensors = segments_tensors.to('cuda')
    # model.to('cuda')
    #
    # # Predict hidden states features for each layer
    # with torch.no_grad():
    #     encoded_layers, _ = model(tokens_tensor, segments_tensors)
    # # We have a hidden states for each of the 12 layers in model bert-base-uncased
    # assert len(encoded_layers) == 12

    def __init__(self, kind):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased').cuda(device=torch.device('cuda'))
        self.model.training = False
        # cache = os.path.join(os.environ.get('CACHE_DIR', os.getcwd()), '.vector_cache')
        # self.glove = torchtext.vocab.GloVe(name=kind, cache=cache)
        self.dim = 768#self.glove.dim
        # self.vectors = self.glove.vectors
        # print("Glove tut")
        self.model.to('cuda')

    @functools.lru_cache(maxsize=1024)
    def tokenize(self, text):
        return self.tokenizer.tokenize(text)

    def untokenize(self, tokens):
        return ' '.join(tokens)

    def lookup(self, token):
        if token not in self.tokenizer.vocab:
            return None

        encoded = self.tokenizer.convert_tokens_to_ids([token])
        return self.model(torch.cuda.LongTensor([encoded]))[0][0].reshape([768])

    def contains(self, token):
        return self.tokenizer.vocab.__contains__(token)

    def to(self, device):
        self.vectors = self.vectors.to(device)
