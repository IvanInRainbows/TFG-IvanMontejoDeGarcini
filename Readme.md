# Description

# Changelog

28/03/2025: Extended the dependency parser model evaluation to include more accurate metrics (precision, recall and f1 score). Given the computational processing power required the code has been adapted to Google Colab, as local runtime takes too long.

25/03/2025: Extended the dependency parser comparison between stanza and SpaCy there's information on the accuracy of stanza with the sentence tokenizer enabled. Extracted comparative and superlative structures for the Lexical features. Extracted the NER count for other features. Designed and extracted the syntactic features of order of constituents by checking the position of the subject and object relative to the verb or whether the subject is implicit. Added some comments for the sake of clarity and legibility.

08/03/2025: Completed the evaluation the accuracy of stanza and SpaCy when parsing syntactic structures. SpaCy's sentence tokenizer cant be disabled inside the Dependency Parser, so there might be some margin of error.

05/03/2025: Managed to make Freeling work and to take the output into a python object as a matrix of dictionaries (list[list[dict]]). The fist dimension constitutes the sentences, the second constitutes each word and each key of the dictionary refers to the following: ID FORM LEMMA TAG SHORT_TAG MSD NEC SENSE SYNTAX DEPHEAD DEPREL COREF SRL. Malt parser has been partially implemented but it depends on the freeling output.

03/03/2025: Started the evaluation of the syntactic parser that will be used for the Syntactic features. At least two parsers will be used (spacy and stanza). The third one might be MaltParser if I manage to make it work. Bash script to parse corpus with MaltParser written, but it doesn't get along with CoNLLu corpus annotation.

02/03/2025: Documented main Dataset class. Added POS bigram matrix DataFrame for each text as an attribute of the class, as well as two functions to initialize and process a bigram matrix given some keys. Added the following POS bigram features: adjective+verb, conjunction+verb, conjunction+noun, conjunction+adjective, conjunction+adverb, pronoun+verb, noun+verb. Also added capital letter count feature as well as some miscellaneous comments.

28/02/2025: Class created for the processing and storage of the dataset. Finished the extraction of the following word count features: number of sentences, number of words, average words per sentence, word character count, average word density, number of sentences with fewer than 12 words and number of sentences with more than 33 words. Extraction of the next lexical features: noun count, verb count, auxiliar verb count, adjective count, adverb count, conjugation count, preposition count and preposition count. Extraction of the punctuation count features.

22/02/2025: Initial commit.
