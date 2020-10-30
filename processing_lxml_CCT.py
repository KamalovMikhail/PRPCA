
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
import os
import numpy as np
import pandas as pd
import tqdm
from sklearn.neighbors import NearestNeighbors


# Datta fo COVID clinical trials available by link
# https://clinicaltrials.gov/ct2/download_studies?term=COVID&down_chunk=2

entries = os.listdir('covid/')

text_ = []
class_= []
for i in entries:
    tt = ""
    with open('covid/'+i, "r") as file:
        content = file.readlines()
        content = "".join(content)
        bs_content = BeautifulSoup(content, "lxml")
        if bs_content.find("official_title" )!= None :
            third_child = bs_content.find("official_title" )
            tt+= list( list(third_child.children))[0]
        if bs_content.find("brief_summary" )!= None :
            third_child = bs_content.find("brief_summary" )
            tt+= ' '.join( list(third_child.strings) ).replace('\n', '')
        if bs_content.find("detailed_description" )!= None :
            third_child = bs_content.find("detailed_description" )
            tt+=' '.join( list(third_child.strings) ).replace('\n', '')
        if bs_content.find("eligibility" )!= None :
            third_child = bs_content.find("eligibility" )
            tt+= ' '.join( list(third_child.strings) ).replace('\n', '')
        if bs_content.find("masking" ) != None:
            class_.append(bs_content.find("masking" ).text)
        else:
            class_.append(0)
        text_.append(tt)

ntext = []
for t in class_:
    if t != 0:
        tt = t.split(' ')[0]
        if tt == 'None':
            tt = t.split(' ')[1].replace("(","")
        ntext.append(tt)
    else:
        ntext.append(0)
d = {'text': text_, 'class':ntext}
df = pd.DataFrame(data=d)
df_free = df[df['class'] != 0]
df_free["class"].replace({"Double": "Blind",
                          "Quadruple": "Blind",
                          "Single":"Blind",
                          "Triple":"Blind", }, inplace=True)
# Features bag-of-words
vectorizer = CountVectorizer( stop_words='english', analyzer='word', min_df=5,)
X = vectorizer.fit_transform(df_free['text'])
# label encoding
le = preprocessing.LabelEncoder()
Y = le.fit_transform(df_free['class'])
# Synthetic A gegneration
neigh = NearestNeighbors(n_neighbors=7, metric='dice', )
neigh.fit(X.toarray())
distances, indices = neigh.kneighbors(X.toarray())
Af = np.zeros((X.shape[0], X.shape[0]))
for i in tqdm.tqdm(range(X.shape[0])):
    Af[i, indices[i]] = 1
    Af[ indices[i], i] = 1

np.save('A', Af)
np.save('Y', Y)
np.save('X', X)