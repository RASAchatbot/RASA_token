from konlpy.tag  import Mecab

tokenizer = Mecab('C:\\mecab\\mecab-ko-dic')

text = "민주가 만든 쿠키 넘 맛있어"

tokenized = tokenizer.morphs(text)
tokenized_pp = []
for t in tokenized:
    if not t == '▁':
        tokenized_pp.append(t.replace('▁', ''))

print(tokenized_pp)

