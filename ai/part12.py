import json
import pandas as pd
import nltk
import re


nltk.download('punkt')


uk_stopwords = {
    'та', 'і', 'й', 'в', 'у', 'на', 'з', 'що', 'це', 'не', 'до', 'як', 'але',
    'за', 'то', 'чи', 'би', 'або', 'із', 'від', 'для', 'так', 'ж', 'же'
}


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in uk_stopwords]
    return " ".join(tokens)


with open("dateR.json", "r", encoding="utf-8") as f:
    data = json.load(f)


for item in data:
    item["ProcessedComment"] = preprocess_text(item.get("Comment", ""))


df = pd.DataFrame(data)
df.to_csv("processed_ukrainian_reviews.csv", index=False, encoding="utf-8-sig")
df.to_json("processed_ukrainian_reviews.json", indent=2, force_ascii=False)

print("✅ Обробка завершена: українські відгуки очищено від стоп-слів.")
