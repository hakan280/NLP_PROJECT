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

#f = codecs.open('../hepsi_v3/data.json', "r", encoding="utf-8")
with open('../hepsi_v3/data.json') as data_file:    
    data = json.load(data_file)
print(type(data))
counter=0
listem=[]
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
    
print counter

cl = NaiveBayesClassifier(listem)
print ("bitti")
print(cl.classify(("berbat")))  # "pos"
print(cl.classify(("mutlaka alin")) )# "neg"