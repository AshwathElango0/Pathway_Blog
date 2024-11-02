from googlesearch import search
from bs4 import BeautifulSoup
import requests
from bs4 import BeautifulSoup

query = search("Python Programming", num_results=1)

for result in query:
    print(result)
    raw_html = requests.get(result).text
    cleantext = BeautifulSoup(raw_html, "lxml").text
    print(cleantext)