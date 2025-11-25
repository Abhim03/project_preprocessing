from preprocessing.utils import log

class CategoryGrouper:
    """
    Group rare categories into 'Other' based on min_frequency threshold.
    Inspired by Hi!ckathon course (artists top-k technique).
    """

    def __init__(self, config):
        self.enable = config.get("categorical_grouper", {}).get("enable", False)
        self.min_freq = config.get("categorical_grouper", {}).get("min_frequency", 20)
        self.replacement = config.get("categorical_grouper", {}).get("replacement", "Other")

    def apply(self, df, categorical_columns):
        if not self.enable:
            return df

        df = df.copy()
        log("Applying categorical grouping...")

        for col in categorical_columns:
            freq = df[col].value_counts()
            rare = freq[freq < self.min_freq].index
            df[col] = df[col].replace(rare, self.replacement)

        log("Categorical grouping complete.")
        return df
