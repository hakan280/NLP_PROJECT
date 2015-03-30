#-*-coding: utf-8 -*-
from textblob.classifiers import NaiveBayesClassifier
import codecs
import json
train = [
    ('I love  this sandwich.', 'pos'),
    ('This is an amazing place!', 'pos'),
    ('I feel very good about these beers.', 'pos'),
    ('This is my best work.', 'pos'),
    ("What an awesome view", 'pos'),
    ('I do not like this restaurant', 'neg'),
    ('I am tired of this stuff.', 'neg'),
    ("I can't deal with this", 'neg'),
    ('He is my sworn enemy!', 'neg'),
    ('My boss is horrible.', 'neg')
]
test = [
    ('The beer was good.', 'pos'),
    ('I do not enjoy my job', 'neg'),
    ("I ain't feeling dandy today.", 'neg'),
    ("I feel amazing!", 'pos'),
    ('Gary is a friend of mine.', 'pos'),
    ("I can't believe I'm doing this.", 'neg')
    ]

counter=0
listem=[]
""" ------------------JSON PARSER-----------
with open('../hepsi_v3/data.json') as data_file:    
    data = json.load(data_file)
print(type(data))

#f = open('hepsi5', 'w')
for i in data:
    comment=(i["reviews"])
    for c in comment:
        #if c["star"]=="5":

        listem.append((c["comment"].strip() , c["star"].decode("utf-8") ))    
            #toFile=(c["star"].encode("utf-8") + "   " + c["comment"].encode("utf-8").strip() + "\n" )
            #f.write(toFile)
            #print toFile
        counter+=1
"""


# .TXT PARSE 
f = codecs.open("train_data/test_data", "r", encoding="utf-8")
for line in f:
    div=line.split()
    star=div[0]
    comment=div[1:]
    comment=" ".join(comment)

    listem.append((comment, star )) 



print counter
cl = NaiveBayesClassifier(listem)
print(cl.classify(("Ürün çok kullanışlı, tavsiye ederim ")))  
print(cl.classify(("hemem bozuldu kalitesiz, aldığıma pişmanım")) )
cl.words()