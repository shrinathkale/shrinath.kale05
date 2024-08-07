import requests
import json
query = input("Enter what type of news you want to see? ")
url = f"https://newsapi.org/v2/everything?q={query}&from=2024-07-07&sortBy=publishedAt&apiKey=a162227e54334b9cbfc69b226731a933"
r = requests.get(url)
news = json.loads(r.text)
for article in news["articles"]:
    print(article["title"])
    print(article["description"])
    print("------------------------------------------------------------------")