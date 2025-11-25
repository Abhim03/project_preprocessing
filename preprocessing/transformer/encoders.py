import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.feature_extraction import FeatureHasher
from category_encoders.target_encoder import TargetEncoder
from preprocessing.utils import log


class Encoders:
    """
    Encodage intelligent basé sur le cours Hi!ckathon 2 :
    
    - OrdinalEncoder : pour variables ordinales uniquement
    - OneHotEncoder : faible cardinalité (nominal)
    - TargetEncoder : moyenne cardinalité (si y dispo)
    - FrequencyEncoding : alternative sans target leakage
    - FeatureHasher : très haute cardinalité
    
    Choix basé sur :
    - cardinalité
    - type détecté par le schema
    - présence/absence de target
    - paramètres dans settings.yaml
    """

    def __init__(self, config):
        self.config = config["encoding"]

        self.onehot_encoders = {}
        self.ordinal_encoders = {}
        self.target_encoders = {}
        self.frequency_maps = {}
        self.hashers = {}

    # ----------------------------------------------------------------------
    # ENCODAGES DE BASE
    # ----------------------------------------------------------------------

    def _encode_onehot(self, df, col):
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        transformed = encoder.fit_transform(df[[col]])

        new_cols = [f"{col}_{cat}" for cat in encoder.categories_[0]]
        df_encoded = pd.DataFrame(transformed, columns=new_cols, index=df.index)

        self.onehot_encoders[col] = encoder
        return df_encoded

    def _encode_ordinal(self, df, col):
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        transformed = encoder.fit_transform(df[[col]])

        df_encoded = pd.DataFrame(transformed, columns=[col], index=df.index)
        self.ordinal_encoders[col] = encoder
        return df_encoded

    def _encode_target(self, df, col, y):
        encoder = TargetEncoder()
        df_encoded = encoder.fit_transform(df[col], y)
        self.target_encoders[col] = encoder
        return df_encoded.rename(col)

    def _encode_frequency(self, df, col):
        freq = df[col].value_counts(normalize=True)
        self.frequency_maps[col] = freq
        return df[col].map(freq).fillna(0)

    def _encode_hashing(self, df, col):
        hasher = FeatureHasher(
            n_features=self.config.get("hashing_features", 10),
            input_type="string"
        )
        hashed = hasher.transform(df[col].astype(str)).toarray()

        new_cols = [f"{col}_hash_{i}" for i in range(hashed.shape[1])]
        df_encoded = pd.DataFrame(hashed, columns=new_cols, index=df.index)

        self.hashers[col] = hasher
        return df_encoded

    # ----------------------------------------------------------------------
    # DECISION LOGIC
    # ----------------------------------------------------------------------

    def _choose_method(self, col, df, schema, y):

        # 1. ORDINAl — priorité
        if schema.get(col) == "ordinal":
            return "ordinal"

        unique = df[col].nunique()
        max_onehot = self.config.get("max_categories_for_onehot", 10)
        high_card = self.config.get("high_cardinality_threshold", 40)

        # 2. OneHot
        if unique <= max_onehot:
            return "onehot"

        # 3. Target Encoding si y dispo
        if max_onehot < unique <= high_card and y is not None:
            return "target"

        # 4. Frequency Encoding si pas de target
        if max_onehot < unique <= high_card and y is None:
            return "frequency"

        # 5. Très haute cardinalité
        return "hashing"

    # ----------------------------------------------------------------------
    # ENCODAGE GLOBAL
    # ----------------------------------------------------------------------

    def apply(self, df, schema, y=None):

        log("Début encodage intelligent...")

        df_encoded = df.copy()

        for col in list(df.columns):
            if schema[col] != "categorical":
                continue

            method = self._choose_method(col, df, schema, y)
            unique = df[col].nunique()

            log(f"Colonne '{col}' — uniques={unique} — méthode={method}")

            if method == "onehot":
                new_cols = self._encode_onehot(df_encoded, col)
                df_encoded = df_encoded.drop(columns=[col])
                df_encoded = pd.concat([df_encoded, new_cols], axis=1)

            elif method == "ordinal":
                df_encoded[col] = self._encode_ordinal(df_encoded, col)

            elif method == "target":
                df_encoded[col] = self._encode_target(df_encoded, col, y)

            elif method == "frequency":
                df_encoded[col] = self._encode_frequency(df_encoded, col)

            elif method == "hashing":
                new_cols = self._encode_hashing(df_encoded, col)
                df_encoded = df_encoded.drop(columns=[col])
                df_encoded = pd.concat([df_encoded, new_cols], axis=1)

        log("Encodage terminé.")
        return df_encoded

    # ----------------------------------------------------------------------
    # TRANSFORM (TEST SET)
    # ----------------------------------------------------------------------

    def transform(self, df):

        log("Encodage du test set avec encodeurs sauvegardés...")
        df_encoded = df.copy()

        # OneHot
        for col, encoder in self.onehot_encoders.items():
            if col not in df_encoded:
                continue

            transformed = encoder.transform(df_encoded[[col]])
            new_cols = [f"{col}_{cat}" for cat in encoder.categories_[0]]
            df_encoded = df_encoded.drop(columns=[col])
            df_encoded = pd.concat(
                [df_encoded,
                 pd.DataFrame(transformed, index=df.index, columns=new_cols)],
                axis=1
            )

        # Ordinal
        for col, encoder in self.ordinal_encoders.items():
            if col in df_encoded:
                df_encoded[[col]] = encoder.transform(df_encoded[[col]])

        # Target
        for col, encoder in self.target_encoders.items():
            if col in df_encoded:
                df_encoded[col] = encoder.transform(df_encoded[col])

        # Frequency encoding
        for col, freq_map in self.frequency_maps.items():
            if col in df_encoded:
                df_encoded[col] = df_encoded[col].map(freq_map).fillna(0)

        # Hashing
        for col, hasher in self.hashers.items():
            if col not in df_encoded:
                continue

            hashed = hasher.transform(df_encoded[col].astype(str)).toarray()
            new_cols = [f"{col}_hash_{i}" for i in range(hashed.shape[1])]
            df_encoded = df_encoded.drop(columns=[col])
            df_encoded = pd.concat(
                [df_encoded,
                 pd.DataFrame(hashed, index=df.index, columns=new_cols)],
                axis=1
            )

        log("Transformation du test set terminée.")
        return df_encoded
