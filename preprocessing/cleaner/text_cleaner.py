import re
import nltk
import string
from nltk.corpus import stopwords
from preprocessing.utils import log

class TextCleaner:
    """
    Clean text columns:
    - lowercase
    - remove punctuation
    - remove numbers
    - remove stopwords
    - remove special characters
    """

    def __init__(self, config):
        self.enable = config.get("text_cleaning", {}).get("enable", False)
        self.remove_punctuation_flag = config.get("text_cleaning", {}).get("remove_punctuation", True)
        self.remove_numbers_flag = config.get("text_cleaning", {}).get("remove_numbers", True)
        self.lowercase_flag = config.get("text_cleaning", {}).get("lowercase", True)
        self.remove_stopwords_flag = config.get("text_cleaning", {}).get("remove_stopwords", True)

        # Load stopwords
        nltk.download("stopwords")
        languages = config.get("text_cleaning", {}).get("stopwords_languages", ["english"])
        self.stopwords = []
        for lang in languages:
            try:
                self.stopwords.extend(stopwords.words(lang))
            except:
                pass

        self.pattern = r'[\[\]()\-:;"\/\.\,\?\!\“\”\’\']'

    def _clean_text(self, text):
        if not isinstance(text, str):
            return text

        # Lowercase
        if self.lowercase_flag:
            text = text.lower()

        # Remove punctuation
        if self.remove_punctuation_flag:
            text = re.sub(self.pattern, " ", text)

        # Remove numbers
        if self.remove_numbers_flag:
            text = re.sub(r'\d+', '', text)

        # Remove stopwords
        if self.remove_stopwords_flag:
            tokens = [w for w in text.split() if w not in self.stopwords]
            text = " ".join(tokens)

        # Remove duplicate spaces
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def apply(self, df, text_columns):
        if not self.enable:
            return df

        log("Applying text cleaning...")
        df = df.copy()

        for col in text_columns:
            df[col] = df[col].astype(str).apply(self._clean_text)

        log("Text cleaning complete.")
        return df
