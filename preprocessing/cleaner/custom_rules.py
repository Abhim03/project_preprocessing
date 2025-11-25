import re
from preprocessing.utils import log

class CustomRulesCleaner:
    """
    Custom rule-based cleaning inspired from course:
    - extract main artist (first in list)
    - remove html tags
    - remove urls
    - remove emojis
    - normalize whitespace
    """

    def __init__(self, config):
        self.enable = config.get("custom_rules", {}).get("enable", False)

    def clean_artists(self, x):
        if not isinstance(x, str):
            return x
        return x.split(";")[0]

    def remove_urls(self, text):
        return re.sub(r'http\S+|www\S+', '', text)

    def remove_html(self, text):
        return re.sub(r'<.*?>', '', text)

    def remove_emojis(self, text):
        emoji_pattern = re.compile(
            "["
            u"\U0001F600-\U0001F64F"  # emoticons  
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs  
            u"\U0001F680-\U0001F6FF"  # transport & map symbols  
            u"\U0001F1E0-\U0001F1FF"  # flags  
            "]+",
            flags=re.UNICODE,
        )
        return emoji_pattern.sub(r'', text)

    def normalize_spaces(self, text):
        return re.sub(r'\s+', ' ', text).strip()

    def apply(self, df, rules):
        if not self.enable:
            return df

        df = df.copy()
        log("Applying custom rule-based cleaning...")

        for col, rule in rules.items():
            if col not in df.columns:
                continue

            if rule == "first_artist":
                df[col] = df[col].astype(str).apply(self.clean_artists)

            elif rule == "remove_urls":
                df[col] = df[col].astype(str).apply(self.remove_urls)

            elif rule == "remove_emojis":
                df[col] = df[col].astype(str).apply(self.remove_emojis)

            elif rule == "normalize_spaces":
                df[col] = df[col].astype(str).apply(self.normalize_spaces)

            elif rule == "clean_all":
                df[col] = (
                    df[col].astype(str)
                    .apply(self.remove_urls)
                    .apply(self.remove_html)
                    .apply(self.remove_emojis)
                    .apply(self.normalize_spaces)
                )

        log("Custom cleaning complete.")
        return df
