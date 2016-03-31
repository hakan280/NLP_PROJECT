#-*-coding: utf-8 -*-
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
from nltk.util import ngrams
from nltk.classify.util import accuracy
import json
import codecs

def parse_txt(file_path):
    listem=[]
    f = codecs.open(file_path, "r", encoding="utf-8")
    for line in f:
	    div=line.split()
	    star=div[0]
	    comment=div[1:]
	    comment=" ".join(comment)
	    listem.append(comment.strip().lower())
	    #listem.append((comment.encode("utf-8").strip(), star.encode("utf-8")) )
			#toFile=(c["star"] + "   " + c["comment"].strip() + "\n" )
			#f.write(toFile)
			#counter+=1
    return listem

comments1=parse_txt("/home/kara/NLP_PROJECT/train_data/BTF_train1")
comments3=parse_txt("/home/kara/NLP_PROJECT/train_data/BTF_train3")
comments5=parse_txt("/home/kara/NLP_PROJECT/train_data/BTF_train5")





def word_feats(words):
	return dict([(word, True) for word in words])

star1 = [(word_feats(f.split()), '1') for f in comments1]
star3= [(word_feats(f.split()), '3') for f in comments3]
star5 = [(word_feats(f.split()), '5') for f in comments5]


print("len star1: "+str(len(star1)))
print("len star3: "+str(len(star3)))
print("len star5: "+str(len(star5)))

print "Number of comments to train: " + str( int(len(star1)*3/4)*3)
print "Number of comments to test: " + str( (len(star1)-int(len(star1)*3/4))*3)



cutoff1=int(len(star1)*3/4)
cutoff3 = int(len(star3)*3/4)
cutoff5 = int(len(star5)*3/4)

trainfeats = star3[:cutoff3]+star1[:cutoff1]+star5[:cutoff5]

testfeats = star3[cutoff3:]+star1[cutoff1:]+star5[cutoff5:]

#print(star3[cutoff3:]+star1[cutoff1:]+star5[cutoff5:])

cl = NaiveBayesClassifier.train(trainfeats)


#print("Test result: " + cl.classify(word_feats("ürünü geçen cuma aldım, çok kaliteli beklentilerimi fazlasıyla karşıladı diyebilirim tavsiye ederim")))
#print("Test result: " + cl.classify(word_feats("ürün çok kalite olmasada idare eder")))


print 'Naive bayes success ratio :', nltk.classify.util.accuracy(cl, testfeats)
#cl.show_most_informative_features()



"""
negids = movie_reviews.fileids('neg')
posids = movie_reviews.fileids('pos')
 
negfeats = [(word_feats(movie_reviews.words(fileids=[f])), 'neg') for f in negids]
posfeats = [(word_feats(movie_reviews.words(fileids=[f])), 'pos') for f in posids]
print negfeats[:1]
negcutoff = len(negfeats)*3/4
poscutoff = len(posfeats)*3/4
 
trainfeats = negfeats[:negcutoff] + posfeats[:poscutoff]
testfeats = negfeats[negcutoff:] + posfeats[poscutoff:]
print 'train on %d instances, test on %d instances' % (len(trainfeats), len(testfeats))
 
classifier = NaiveBayesClassifier.train(trainfeats)
print 'accuracy:', nltk.classify.util.accuracy(classifier, testfeats)
classifier.show_most_informative_features()

"""

from nltk.classify import MaxentClassifier
me_classifier = MaxentClassifier.train(trainfeats, trace=0, max_iter=1, min_lldelta=0.5)
print("Maxent Classifier success rate: " + str(accuracy(me_classifier, testfeats)))


from nltk.classify import DecisionTreeClassifier

dt_classifier = DecisionTreeClassifier.train(trainfeats,binary=True, entropy_cutoff=0.8, depth_cutoff=5, support_cutoff=30)

print("Decision Tree classifier success rate: "+ str(accuracy(dt_classifier, testfeats)))

print cl.show_most_informative_features()