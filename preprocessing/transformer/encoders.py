import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.feature_extraction import FeatureHasher
from category_encoders.target_encoder import TargetEncoder
from preprocessing.utils import log


class Encoders:
    """
    Applique l'encodage des variables catégorielles selon :
    - faible cardinalité   -> OneHot
    - moyenne cardinalité  -> Target Encoding
    - haute cardinalité    -> Hashing
    - ordinales            -> OrdinalEncoder

    Le choix dépend :
    - du type détecté (schema)
    - du nombre de valeurs uniques
    - des paramètres dans settings.yaml
    """

    def __init__(self, config):
        self.config = config["encoding"]

        # Stockage des encodeurs pour pouvoir réutiliser sur train/test
        self.encoders = {}
        self.hashers = {}
        self.ordinal_encoders = {}
        self.onehot_encoders = {}
        self.target_encoders = {}

    # ----------------------------------------------------------------------
    # Méthodes d'encodage
    # ----------------------------------------------------------------------

    def _encode_onehot(self, df, col):
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        transformed = encoder.fit_transform(df[[col]])

        new_cols = [f"{col}_{cat}" for cat in encoder.categories_[0]]

        df_encoded = pd.DataFrame(transformed, columns=new_cols, index=df.index)

        self.onehot_encoders[col] = encoder

        return df_encoded

    def _encode_target(self, df, col, y):
        encoder = TargetEncoder()
        df_encoded = encoder.fit_transform(df[col], y)

        self.target_encoders[col] = encoder

        return df_encoded.rename(col)

    def _encode_hashing(self, df, col):
        hasher = FeatureHasher(
            n_features=10, input_type="string"
        )  # 10 features par défaut

        hashed = hasher.transform(df[col].astype(str)).toarray()

        new_cols = [f"{col}_hash_{i}" for i in range(hashed.shape[1])]
        df_encoded = pd.DataFrame(hashed, columns=new_cols, index=df.index)

        self.hashers[col] = hasher

        return df_encoded

    def _encode_ordinal(self, df, col):
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        transformed = encoder.fit_transform(df[[col]])

        df_encoded = pd.DataFrame(transformed, columns=[col], index=df.index)

        self.ordinal_encoders[col] = encoder
        return df_encoded

    # ----------------------------------------------------------------------
    # Méthode principale : appliquer l’encodage
    # ----------------------------------------------------------------------

    def apply(self, df, schema, y=None):
        """
        df : DataFrame à encoder
        schema : dict types des colonnes
        y : target pour target encoding (si disponible)

        Retour :
        df transformé avec encodage correct
        """

        log("Début encodage des colonnes catégorielles...")

        df_encoded = df.copy()
        cat_threshold = self.config["high_cardinality_threshold"]

        for col in df.columns:

            if schema[col] != "categorical":
                continue

            unique_count = df[col].nunique()

            # Décision de la méthode
            if unique_count <= 10:
                method = "onehot"

            elif 10 < unique_count <= cat_threshold:
                method = "target"

            else:
                method = "hashing"

            # Encodage selon méthode
            if method == "onehot":
                new_cols = self._encode_onehot(df_encoded, col)
                df_encoded = df_encoded.drop(columns=[col])
                df_encoded = pd.concat([df_encoded, new_cols], axis=1)

            elif method == "target":
                if y is None:
                    log(f"Target absente -> fallback en ordinal pour la colonne {col}")
                    new_col = self._encode_ordinal(df_encoded, col)
                else:
                    new_col = self._encode_target(df_encoded, col, y)

                df_encoded[col] = new_col

            elif method == "hashing":
                new_cols = self._encode_hashing(df_encoded, col)
                df_encoded = df_encoded.drop(columns=[col])
                df_encoded = pd.concat([df_encoded, new_cols], axis=1)

            log(f"Colonne '{col}' encodée par : {method}")

        log("Encodage terminé.")

        return df_encoded

    # ----------------------------------------------------------------------
    # Méthode pour transformer test set avec les mêmes encodeurs
    # ----------------------------------------------------------------------

    def transform(self, df):
        """
        Applique les mêmes encodages que sur le train.
        """
        log("Transformation test set avec encodeurs sauvegardés...")

        df_encoded = df.copy()

        # OneHot
        for col, encoder in self.onehot_encoders.items():
            if col not in df_encoded:
                continue
            transformed = encoder.transform(df_encoded[[col]])
            new_cols = [f"{col}_{cat}" for cat in encoder.categories_[0]]
            df_encoded = df_encoded.drop(columns=[col])
            df_encoded = pd.concat(
                [df_encoded, pd.DataFrame(transformed, index=df.index, columns=new_cols)],
                axis=1
            )

        # Target encoding
        for col, encoder in self.target_encoders.items():
            if col in df_encoded:
                df_encoded[col] = encoder.transform(df_encoded[col])

        # Hashing
        for col, hasher in self.hashers.items():
            if col not in df_encoded:
                continue
            hashed = hasher.transform(df_encoded[col].astype(str)).toarray()
            new_cols = [f"{col}_hash_{i}" for i in range(hashed.shape[1])]
            df_encoded = df_encoded.drop(columns=[col])
            df_encoded = pd.concat(
                [df_encoded, pd.DataFrame(hashed, index=df.index, columns=new_cols)],
                axis=1
            )

        # Ordinal
        for col, encoder in self.ordinal_encoders.items():
            if col in df_encoded:
                df_encoded[[col]] = encoder.transform(df_encoded[[col]])

        log("Encodage du test set terminé.")

        return df_encoded
