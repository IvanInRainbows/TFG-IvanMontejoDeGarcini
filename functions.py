import pandas as pd
import numpy as np
import stanza
import re
import itertools
import numpy as np
import spacy
from stanza.utils.conll import CoNLL
#/home/ivan/Desktop/Uni/TFG/RepositorioTFG/Datasets/IULA/IULA_Spanish_LSP_Treebank.conll

class Dataset(pd.DataFrame):
    nlp_stanza = stanza.Pipeline(lang='es', processors='tokenize,ner,mwt,pos,lemma,depparse')
    nlp_spacy = spacy.load("es_core_news_md")
    uposTags = ("ADJ","ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "ADJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X")
    def __init__(self, source : pd.DataFrame):
        """Main object that generates the dataset from a pandas DataFrame object consisting of two columns. The first column should be named 'text' and contain the raw texts. The second column should be named 'label' and contain the label 'human' if the text is human generated or anything else if it's AI generated.

        :param source: The source of the data with columns 'text' and 'label'
        :type source: pd.DataFrame
        """
        super().__init__()
        self["text"] = source["text"]
        self.docs = [Dataset.nlp_stanza(i) for i in self["text"]]
        self.sentences = [i.sentences for i in self.docs]
        self.words = [[j.text for j in i.iter_words()] for i in self.docs]
        self.tokens = [[j.text for j in i.iter_tokens()] for i in self.docs]
        self.lemmas = [[j.lemma for j in i.iter_words()] for i in self.docs]
        self.pos = [[j.upos for j in i.iter_words()] for i in self.docs]
        self.posBigramMatrix = [bigramMatrix(i, self.uposTags) for i in self.pos]
        self["label"] = [0 if i == "human" else 1 for i in source["label"]]
        self.__setPunctuationFeatures()
        self.__setCharWordcountFeatures()
        self.__setLexicalFeatures()
        self.__setBigramFeatures()
    
    def __setPunctuationFeatures(self):
        punct2count = list(".,:;()¿?¡!-\"\\@#$€*+")
        for i in punct2count:
            self[i] = [j.count(i) for j in self["text"]]

    def __setCharWordcountFeatures(self):
        self["sentences"] = [len(i.sentences) for i in self.docs]
        #self["tokens"] = [i.num_tokens for i in self.docs]
        self["words"] = [i.num_words for i in self.docs]
        self["avgWordsPerSent"] = [self["words"][i]/self["sentences"][i] for i in range(len(self.docs))]
        self["wordCharCount"] = [len("".join(i)) for i in self.words]
        self["avgWordDensity"] = [self["wordCharCount"][i]/self["words"][i] for i in range(len(self["words"]))]
        self["nSentsle11Words"] = [sum([1 if len(j.words) <= 11 else 0 for j in i]) for i in self.sentences]
        self["nSentsge34Words"] = [sum([1 if len(j.words) >= 11 else 0 for j in i]) for i in self.sentences]
        self["caps"] = [len(re.findall(r"[A-ZÁÉÍÓÚÑ]", i))for i in self["text"]]
        #TODO: Lexical density, ILFW

    def __setLexicalFeatures(self):
        posTags = ("NOUN", "VERB", "AUX", "ADJ", "PRON", "ADV", "CCONJ", "SCONJ", "ADP", "PROPN", "NUM")
        for i in posTags:
            self["count"+i] = [j.count(i)/len(j) for j in self.pos]
        #TODO: Comparative and superlative adjectives, lexical complexity
    
    def __setBigramFeatures(self):
        self["ADJ+VERB"] = [i["ADJ"]["VERB"] for i in self.posBigramMatrix]
        self["CONJ+VERB"] = [i["CCONJ"]["VERB"]+i["SCONJ"]["VERB"] for i in self.posBigramMatrix]
        self["CONJ+ADJ"] = [i["CCONJ"]["ADJ"]+i["SCONJ"]["ADJ"] for i in self.posBigramMatrix]
        self["CONJ+NOUN"] = [i["CCONJ"]["NOUN"]+i["SCONJ"]["NOUN"] for i in self.posBigramMatrix]
        self["CONJ+ADV"] = [i["CCONJ"]["ADV"]+i["SCONJ"]["ADV"] for i in self.posBigramMatrix]
        self["PRON+VERB"] = [i["PRON"]["VERB"] for i in self.posBigramMatrix]
        self["NOUN+VERB"] = [i["NOUN"]["VERB"] for i in self.posBigramMatrix]
        #TODO: TF-IDF bigram

    #TODO: syntactic structure Features, legibility features, pragmatic/discourse features, Other features

    def debug(self):
        for i in self.sentences:
            for j in i:
                for token in self.nlp_spacy(j.text):
                    print(token.text, token.dep_)
                print(*[f'id: {word.id}\tword: {word.text}\thead id: {word.head}\thead: {j.words[word.head-1].text if word.head > 0 else "root"}\tdeprel: {word.deprel}' for word in j.words], sep='\n')


def bigramMatrix(l : list, keys: list):
    """Given a list of items and the keys (i.e. Text already segmented in words and the unique words) creates and return a bigram count matrix.

    :param l: Input in list format, for example a list of the words POS or segmented text.
    :type l: list
    :param keys: The keys for the bigrams, such as the vocabulary for a word bigram matrix or POS tags for a POS bigram matrix.
    :type keys: list
    :return: Returns the bigram matrix as dictionaries nested in another dictionary with the given keys as keys and the count of the bigram as values of the nested dictionaries.
    :rtype: dict[dict]
    """
    out = initMatrixAsDict(keys)
    for i in range(len(l)-1):
        out[l[i]][l[i+1]]+=1
    return out

def initMatrixAsDict(keys):
    out = {}
    for i in itertools.product(keys, keys):
        if i[0] in out.keys() and out[i[0]] != None:
            out[i[0]].update({i[1]:0})
        else:
            out[i[0]] = {i[1]:0}
    return out

if __name__ == "__main__":
    print("Hola")