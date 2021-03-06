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



# ------------------JSON PARSER-----------
def parse_json(json_path,txt_name):
	#'../hepsi_v3/data.json'
	with open(json_path) as data_file:    
		data = json.load(data_file)
		print(type(data))
	counter=0
	f = open(txt_name, 'w')
	for i in data:
		comment=(i["reviews"])
		for c in comment:
			if c["star"]=="5":

				#listem.append((c["comment"].strip() , c["star"].decode("utf-8") ))    
				toFile=(c["star"].encode("utf-8") + "   " + c["comment"].encode("utf-8").strip() + "\n" )
				f.write(toFile)
			
				counter+=1

parse_json('../hepsi_v3/data.json',"tablet_comments5")

# .TXT PARSE 
def parse_txt(path):
	listem=[]
	f = codecs.open(path, "r", encoding="utf-8")
	for line in f:
		div=line.split()
		star=div[0]
		comment=div[1:]
		comment=" ".join(comment)
		listem.append((comment.lower(), star )) 
	return listem

train_list=parse_txt("train_data/train_data")
test_list=parse_txt("train_data/test_data")
t45=parse_txt("train_data/45")
print len(test_list)

cl = NaiveBayesClassifier(train_list)
cl2=NaiveBayesClassifier(t45)
def find_star(comment):
	star1=cl.classify(comment)

	if star1=="5":
		star2=cl2.classify(comment)
		return star2



"""


print(cl.classify((u"Ürün çok kullanışlı, tavsiye ederim ")))  
print(cl.classify((u"hemem bozuldu kalitesiz, aldığıma pişmanım")) )
print(cl.classify((u"Ürün iyi ama tek eksiği şarjı az gidiyor, fiyatına göre idare eder")))  
print(cl.classify(U"kötü"))  
print(cl.classify((u"Müthiş, görüntü kalitesi mükemmel.")))  
print(cl.classify((u"Verdiğiniz paraya değer tavsive ederim ")))  
print(cl.classify((u"çok dandik, işe yaramaz"))) 
print(cl.classify((u"kalitesiz")))
print(cl.classify((u"ürün kaliteli")))    
"""
