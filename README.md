# fear
study of fear in conspiracy theory discussion communities

### to install
python requirements and resources:
```shell
pip install -r requirements.txt
python -m nltk.downloader punkt
python -m nltk.downloader stopwords
python -m nltk.downloader vader_lexicon
python -m spacy download en_core_web_lg
``` 

### to run analyses
* scrape_fear.py: find expressions of fear on reddit, using pushshift