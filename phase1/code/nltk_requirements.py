import nltk

with open("nltk_downloads.txt") as f:
    for line in f:
        nltk.download(line.strip())
