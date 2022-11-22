import ko_core_news_sm
nlp = ko_core_news_sm.load()
doc = nlp("내 이름은 홍길동입니다.")
print([(w.text, w.pos_) for w in doc])