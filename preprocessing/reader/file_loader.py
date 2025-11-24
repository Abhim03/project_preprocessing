import os
import pandas as pd
from preprocessing.utils import safe_read_csv, log


class FileLoader:
    """
    Module responsable de charger un fichier de données
    quel que soit son format (csv, excel, parquet, json).
    """

    SUPPORTED_FORMATS = ["csv", "xlsx", "xls", "parquet", "json"]

    def __init__(self):
        pass

    def _detect_extension(self, path):
        """Retourne l'extension d'un fichier."""
        if "." not in path:
            raise ValueError("Aucune extension détectée dans le fichier : " + path)
        return path.split(".")[-1].lower()

    def load(self, path):
        """Charge un fichier en fonction de son extension."""

        if not os.path.exists(path):
            raise FileNotFoundError(f"Le fichier '{path}' est introuvable")

        ext = self._detect_extension(path)

        if ext not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Format '{ext}' non supporté. Formats supportés : {self.SUPPORTED_FORMATS}"
            )

        log(f"Chargement du fichier '{path}' (format : {ext})")

        if ext == "csv":
            df = safe_read_csv(path)

        elif ext in ["xlsx", "xls"]:
            df = pd.read_excel(path)

        elif ext == "parquet":
            df = pd.read_parquet(path)

        elif ext == "json":
            df = pd.read_json(path)

        else:
            raise ValueError("Extension du fichier non reconnue.")

        log(f"Fichier chargé avec succès. Dimensions : {df.shape[0]} lignes, {df.shape[1]} colonnes")

        return df
