import json

import requests
import urllib.parse

data = [
    "Find me employees in Asia.",
    "Find expert who are in the city Chicago.",
    "Search experts are in New York City region.",
    "I am looking for an expert in the vicinity of South America.",
    "I am looking for an expert located in London.",
    "BMW's experts.",
    "Find me experts from BMW.",
    "I am looking for an expert that is working on Mercedes account.",
    "Find me an expert that has experience with Toyota client.",
    "Find me an employee that has experience with Mercedes client.",
    "Locate someone with expertise in cartoon.",
    "Search for an expert in Web design.",
    "Find an expert in web page design.",
    "I am looking for an expert who's interest in retail.",
    "Find an expert with experience in animation located in Chicago.",
    "Find an expert who is interest in big data located in South  America.",
    "which experts are near me.",
    "Find an expert have skill in machine learning located in North America.",
    "Show me employees who are in New York City and interest in big data.",
    "Show me employees who are in New York City and have experience with web design.",
    "Find me employees who know machine learning.",
    "Find me employees who have experience with web design.",
    "Show me employee who are in Asia and have worked in Toyota's projects.",
    "Find me employees who have experience with web design and interest in big data.",
    "Find me employees who work in Audi's projects.",
    "Find me employees who work for Audi's account.",
    "Show me employees who are interested in big data.",
    "Show me employees who is interested in big data and in London.",
    "Find me employees who have skill in big data.",
    "Find me an active full time logo artist who are from brand communication department and interest in big data and baseball.",
    "Find Arpena profile.",
    "Show me Veronique profile.",
    "Tell me more about Michael Smith.",
    "Look up Michael Smith.",
    "View employees profile.",
    "Michael Smith's profile detail.",
    "Find me someone similar to Amanda Brown.",
    "Find me an designer .",
    "Is there an designer in Amanda Brown office.",
    "Find me an designer with experience in website.",
    "Find me someone who speaks Japanese .",
    "Someone with experience in mobile and integrated campaign.",
    "Find me an designer for a Pitch.",
    "Find me someone with experience in mobile and integrated campaign.",
    "Find me someone who might be interested in cartoon work.",
    "Find me an designer, product manager, and web developer.",
    "Someone with data science and programming.",
    "Someone who's great at data science and programming.",
    "Someone with website experience.",
    "Find me an designer, product manager, and web developer with website experience.",
    "Someone who know machine learning or pattern recognition.",
    "Get me employees who have experience on mobile or integrated campaign.",
]

param = {"annotators": "tokenize,ssplit,truecase,pos,depparse,ner,entitymentions", "outputFormat": "json"}
param = urllib.parse.quote(json.dumps(param))
url = "http://c.maochen.org:9000/?properties=" + param

for d in data:
    resp = requests.post(url=url, data=d)
    resp = json.loads(resp.text)
    true_sent = ""
    for token in resp["sentences"][0]["tokens"]:
        is_tagged = False
        if token["ner"] != "O":
            true_sent += "<E ner=\"" + token["ner"] + "\">"
            is_tagged = True
        true_sent += token["truecaseText"]
        if is_tagged:
            true_sent += "<\\E>"
        true_sent += token["after"]

    print(true_sent)
