#%%
import MeCab
mecab = MeCab.Tagger()
# #out = mecab.parse("오늘은 맑은 날씨이다.")
# out = mecab.parse("Microsoft Corporation. All rights reserved.")
# print(out)

wakati = MeCab.Tagger("-Owakati")
out = wakati.parse("Microsoft Corporation. All rights reserved.").split()
print(out)
l1 = ''
for l in out: l1 += l+' '

print(l1)
#print(out)