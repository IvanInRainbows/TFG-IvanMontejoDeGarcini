import pandas as pd
import numpy as np
import stanza

class Dataset(pd.DataFrame):
    nlp_stanza = stanza.Pipeline(lang='es', processors='tokenize,ner,mwt,pos,lemma')
    def __init__(self, source : pd.DataFrame):
        super().__init__()
        self["text"] = source["text"]
        self.__docs = [Dataset.nlp_stanza(i) for i in self["text"]]
        self.__sentences = [i.sentences for i in self.__docs]
        self.__words = [[j.text for j in i.iter_words()] for i in self.__docs]
        self.__tokens = [[j.text for j in i.iter_tokens()] for i in self.__docs]
        self.__lemmas = [[j.lemma for j in i.iter_words()] for i in self.__docs]
        self.__pos = [[j.upos for j in i.iter_words()] for i in self.__docs]
        self["label"] = [0 if i == "human" else 1 for i in source["label"]]
        self.__setPunctuationFeatures()
        self.__setCharWordcountFeatures()
        self.__setLexicalFeatures()
    
    def __setPunctuationFeatures(self):
        punct2count = list(".,:;()¿?¡!-\"\\@#$€*+")
        for i in punct2count:
            self[i] = [j.count(i) for j in self["text"]]

    def __setCharWordcountFeatures(self):
        self["sentences"] = [len(i.sentences) for i in self.__docs]
        #self["tokens"] = [i.num_tokens for i in self.__docs]
        self["words"] = [i.num_words for i in self.__docs]
        self["avgWordsPerSent"] = [self["words"][i]/self["sentences"][i] for i in range(len(self.__docs))]
        self["wordCharCount"] = [len("".join(i)) for i in self.__words]
        self["avgWordDensity"] = [self["wordCharCount"][i]/self["words"][i] for i in range(len(self["words"]))]
        self["nSentsle11Words"] = [sum([1 if len(j.words) <= 11 else 0 for j in i]) for i in self.__sentences]
        self["nSentsge34Words"] = [sum([1 if len(j.words) >= 11 else 0 for j in i]) for i in self.__sentences]

    def __setLexicalFeatures(self):
        posTags = ("NOUN", "VERB", "AUX", "ADJ", "PRON", "ADV", "CCONJ", "SCONJ", "ADP", "PROPN")
        for i in posTags:
            self["count"+i] = [j.count(i)/len(j) for j in self.__pos]

    def debug(self):
        for i in self.__docs:
            for sent in i.sentences:
                for w in sent.words:
                    print(f"{w.text} : {w.upos} or {w.xpos}")

if __name__ == "__main__":
    a = Dataset(pd.read_excel("/home/ivan/Desktop/Uni/TFG/RepositorioTFG/Datasets/out.xlsx"))
    print(a)