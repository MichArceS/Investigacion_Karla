import gensim
from gensim import corpora
from pprint import pprint
import pandas as pd
import numpy as np
import json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

stop_words = set(stopwords.words('spanish'))

_FILE = "data.csv"

df = pd.read_csv(_FILE, dtype=str)
_file_tmp = open("questions.json", encoding="utf8")
questions = json.load(_file_tmp)

matrix = []
for column in df.columns[1:]:
    matrix.append(list(df[column]))

final_dict = {}
for i in range(0, len(questions.keys())):
    q = list(questions.keys())[i]
    _data = matrix[i]
    tokens = [gensim.utils.simple_preprocess(d) for d in _data]
    textos_filtrados = [[palabra for palabra in texto if palabra not in stop_words] for texto in tokens]
    if len(textos_filtrados[0]) == 0:
        print("pass")
    else:
        _dict = corpora.Dictionary(textos_filtrados)
        corpus = [_dict.doc2bow(text) for text in textos_filtrados]
        lda_model = gensim.models.LdaModel(corpus, num_topics=5, id2word=_dict, passes=100)
        final_dict[q] = lda_model.print_topics()
json_object = json.dumps(final_dict)
with open("topic_modeling.json", "w") as outfile:
    outfile.write(json_object)
