from preprocessing.utils import log


class DuplicatesHandler:
    """
    Gère les doublons dans le dataset.
    Fonctionnalités :
    - Détection du nombre de doublons
    - Suppression si l'option 'remove' est activée dans settings.yaml
    """

    def __init__(self, config):
        self.config = config["duplicates"]

    def analyze(self, df):
        """
        Retourne le nombre de doublons détectés.
        """
        duplicate_count = df.duplicated().sum()
        log(f"Nombre de doublons détectés : {duplicate_count}")
        return duplicate_count

    def remove(self, df):
        """
        Supprime les doublons si activé dans la configuration.
        """
        if not self.config["remove"]:
            log("Suppression des doublons désactivée dans les paramètres.")
            return df

        before = len(df)
        df_clean = df.drop_duplicates()
        after = len(df_clean)

        removed = before - after
        log(f"Doublons supprimés : {removed}")

        return df_clean
