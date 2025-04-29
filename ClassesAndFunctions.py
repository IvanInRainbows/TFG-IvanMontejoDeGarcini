import pandas as pd
import numpy as np
import stanza
import re
import itertools
from stanza.utils.conll import CoNLL
import textstat
import language_tool_python as lt
textstat.set_lang("es")


class Dataset(pd.DataFrame):
    nlp_stanza = stanza.Pipeline(lang='es', processors='tokenize,ner,mwt,pos,lemma,depparse', use_gpu=True)
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
        self.nChars = [len(i) for i in self["text"]]
        self["label"] = [0 if i == "human" else 1 for i in source["label"]]
        self.__setPunctuationFeatures()
        self.__setCharWordcountFeatures()
        self.avgWordsPerSentComparison = {
            'human': np.mean([self["avgWordsPerSent"][i] for i in range(len(self["label"])) if self["label"][i] == 0]),
            'gpt': np.mean([self["avgWordsPerSent"][i] for i in range(len(self["label"])) if self["label"][i] == 1])
        }
        self.__setLexicalFeatures()
        self.__setBigramFeatures()
        self.__setSyntacticFeatures()
        self.__setOtherFeatures()
        self.__readabilityComplexFeatures()
    
    def __setPunctuationFeatures(self):
        punct2count = list(".,:;()¿?¡!-\"\\@#$€*+")
        for i in punct2count:
            self[i] = [self["text"][j].count(i)/len(self.tokens[j]) for j in range(len(self["text"]))]

    def __setCharWordcountFeatures(self):
        self["sentences"] = [len(i.sentences) for i in self.docs]
        #self["tokens"] = [i.num_tokens for i in self.docs]
        self["words"] = [i.num_words for i in self.docs]
        self["TTR"] = np.array([len(set([j.text for j in i.iter_words()])) for i in self.docs])/self["words"]
        self["avgWordsPerSent"] = [self["words"][i]/self["sentences"][i] for i in range(len(self.docs))]
        self["wordCharCount"] = [len("".join(self.words[i]))/self.nChars[i] for i in range(len(self.words))]
        self["avgWordDensity"] = [self["wordCharCount"][i]/self["words"][i] for i in range(len(self["words"]))]
        self["nSentsl16Words"] = [sum([1 if len(j.words) <= 15 else 0 for j in i])/len(i) for i in self.sentences]
        self["nSentsg26Words"] = [sum([1 if len(j.words) >= 25 else 0 for j in i])/len(i) for i in self.sentences]
        #self["caps"] = [len(re.findall(r"[A-ZÁÉÍÓÚÑ]", i)) for i in self["text"]] # This feature proved not to be relevant for the corpus
        self["Lexical density"] = [len([j.text for j in self.docs[i].iter_words() if j.pos in ("ADJ", "ADV", "VERB", "NOUN")])/len(self.words[i]) for i in range(len(self.docs))]
        _tmp = [set([j.lemma for j in i.iter_words() if j.pos in ("ADJ", "ADV", "VERB", "NOUN")]) for i in self.docs]
        self["Lexical diversity"] = [len(_tmp[i])/len(self.sentences[i]) for i in range(len(_tmp))]
        del _tmp

    def __setSyntacticFeatures(self):
        """This checks the position of the object and the subject relative to the verb or whether the subject is ommited"""
        out = np.ndarray(shape = (0, 6))
        for doc in self.docs:
            order = [0] * 6 # Indexes: 0=Subject-Verb, 1=Verb-Subject, 2=No explicit subject, 3=Verb-Object, 4=Object-Verb, 5=obl-Verb

            for i in range(len(doc.sentences)): 
                current = []

                for j in range(len(doc.sentences[i].words)):
                    # For each word check if its the deprel that we are looking for. Also checks for the subject of subordinate clauses.
                    if doc.sentences[i].words[j].deprel in ("csubj", "nsubj", "root", "obj", "obl", "obl:arg"):
                        current.append(doc.sentences[i].words[j].deprel) # Here the consituent order is stored to check whether a verb has already appeared or not.
                        match doc.sentences[i].words[j].deprel:
                            case "nsubj" | "csubj":
                                if "root" not in current:
                                    order[0]+=1
                                else:
                                    order[1]+=1
                            case "obj":
                                if "root" not in current:
                                    order[4]+=1
                                else:
                                    order[3]+=1
                            case "obl" | "obl:arg":
                                if "root" not in current:
                                    order[5]+=1

                    elif doc.sentences[i].words[j].deprel == "cc" or (doc.sentences[i].words[j].deprel == "punc" and "VERB" in [i.pos for i in doc.sentences[i].words[j:j+5]]): # Subordinate sentences can be a problem. Thus when a conjunction or sentence connector is found we assume that a new sentence starts. A very similar thing happens if we find a coma followed by a verb in one of the next five words. Nonetheless, this is not a perfect solution.
                        if "nsubj" not in current and "csubj" not in current and "root" in current: # Implicit or ommited subjects
                            order[2]+=1
                        current = []  

                if "nsubj" not in current and "csubj" not in current and "root" in current: # Implicit or ommited subjects
                    order[2]+=1

            out = np.insert(out, len(out), order, axis=0)

        out = np.transpose(out)# As each nested array contains the information about one review transposing the matrix means that now every nested array contains one of the count of one oof the positions of the subject/object
        self.nSents = np.array([len(i) for i in self.sentences])
        labels = ["Subject-Verb","Verb-Subject","No explicit subject","Verb-Object","Object-Verb", "obl-Verb"]
        for i in range(len(out)):
            self[labels[i]] = out[i]/self.nSents

    def __setLexicalFeatures(self):
        posTags = ("NOUN", "VERB", "AUX", "ADJ", "PRON", "ADV", "CCONJ", "SCONJ", "ADP", "PROPN", "NUM")
        for i in posTags:
            self["count"+i] = [j.count(i)/len(j) for j in self.pos]
        #Comparatives and superlatives
        _comparatives = []
        _superlatives = []
        for doc in self.docs:
            comparatives = 0
            superlatives = 0
            for sent in doc.sentences:
                for i in range(len(sent.words)-1):
                    # The comprobation is made relative to the comparative adverb, first check if it's a superlative and then, in case it's not, check comparative.
                    if i > 0 and sent.words[i-1].text.lower() in ("el", "la", "lo", "los", "las") and sent.words[i].text.lower() in ("mas", "más", "menos") and sent.words[i+1].pos in ("ADJ", "ADV"):
                        superlatives += 1
                    elif sent.words[i].text.lower() in ("mas", "más", "menos") and sent.words[i+1].pos in ("ADJ", "ADV"):
                        comparatives+=1
            _comparatives.append(comparatives)
            _superlatives.append(superlatives)
        self["Comparatives"] = np.array(_comparatives)/self["words"]
        self["Superlatives"] = np.array(_superlatives)/self["words"]
        del _comparatives, _superlatives
    
    def __setBigramFeatures(self):
        # Makes POS bigram count from the POS bigram matrix attribute
        self["ADJ+VERB"] = [i["ADJ"]["VERB"] for i in self.posBigramMatrix]
        self["CONJ+VERB"] = [i["CCONJ"]["VERB"]+i["SCONJ"]["VERB"] for i in self.posBigramMatrix]
        self["CONJ+ADJ"] = [i["CCONJ"]["ADJ"]+i["SCONJ"]["ADJ"] for i in self.posBigramMatrix]
        self["CONJ+NOUN"] = [i["CCONJ"]["NOUN"]+i["SCONJ"]["NOUN"] for i in self.posBigramMatrix]
        self["CONJ+ADV"] = [i["CCONJ"]["ADV"]+i["SCONJ"]["ADV"] for i in self.posBigramMatrix]
        self["PRON+VERB"] = [i["PRON"]["VERB"] for i in self.posBigramMatrix]
        self["NOUN+VERB"] = [i["NOUN"]["VERB"] for i in self.posBigramMatrix]
        # NORMALIZATION
        self["ADJ+VERB"]/=self["words"]
        self["CONJ+VERB"]/=self["words"]
        self["CONJ+ADJ"]/=self["words"]
        self["CONJ+NOUN"]/=self["words"]
        self["CONJ+ADV"]/=self["words"]
        self["PRON+VERB"]/=self["words"]
        self["NOUN+VERB"]/=self["words"]

    def __setOtherFeatures(self):
        self["NERS"] = [len(doc.entities)/doc.num_words for doc in self.docs]
        tool = lt.LanguageTool('es')
        self["Grammar errors"] = [len(tool.check(t)) for t in self["text"]]
        self["Grammar errors"] /= self["words"]

    def __readabilityComplexFeatures(self):
        self["G. Polini"] = [textstat.gutierrez_polini(t) for t in self["text"]]
        self["F. Huerta readability"] = [textstat.fernandez_huerta(t) for t in self["text"]]
        self["Crawford score"] = [textstat.crawford(t) for t in self["text"]]

    def show_wordspersent(self):
        print(f"Average words per sentence in human texts: {self.avgWordsPerSentComparison['human']}")
        print(f"Average words per sentence in human texts: {self.avgWordsPerSentComparison['gpt']}")


def bigramMatrix(l : list, keys: list):
    """Given a list of items and the keys (i.e. Text already segmented in words and the unique words) creates and return a bigram count matrix.

    :param l: Input in list format, for example a list of the words POS or segmented text.
    :type l: list
    :param keys: The keys for the bigrams, such as the vocabulary for a word bigram matrix or POS tags for a POS bigram matrix.
    :type keys: list
    :return: Returns the bigram matrix as dictionaries nested in another dictionary with the given keys as keys and the count of the bigram as values of the nested dictionaries.
    :rtype: dict[dict]
    """
    out = initMatrixAsDict(keys, keys)
    for i in range(len(l)-1):
        out[l[i]][l[i+1]]+=1
    return out

def initMatrixAsDict(keys_x, keys_y):
    out = {}
    for i in itertools.product(keys_x, keys_y):
        if i[0] in out.keys() and out[i[0]] != None:
            out[i[0]].update({i[1]:0})
        else:
            out[i[0]] = {i[1]:0}
    return out

if __name__ == "__main__":
    pass