from preprocessing.pipeline import PreprocessingPipeline

pipe = PreprocessingPipeline()

# Exemple : fichier CSV dans data/
X_train, X_test, y_train, y_test = pipe.run(
    "data/mon_fichier.csv",
    target_col="target"
)
