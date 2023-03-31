import nltk

hypothesis = ['It', 'is', 'a', 'cat', 'at', 'room']
reference = ['It', 'is', 'a', 'cat', 'inside', 'the', 'room']

#there may be several references
BLEUscore = nltk.translate.bleu_score.sentence_bleu([reference], hypothesis)
print(BLEUscore)

#%%
from nltk.translate.bleu_score import sentence_bleu
reference = [['the', 'cat',"is","sitting","on","the","mat"]]
test = ["on",'the',"mat","is","a","cat"]
score = sentence_bleu(  reference, test)
print(score)

#%%

from nltk.translate.bleu_score import sentence_bleu
reference = [['the', 'cat',"is","sitting","on","the","mat"]]
test = ["there",'is',"cat","sitting","cat"]
score = sentence_bleu(  reference, test)
print(score)

