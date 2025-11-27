# api_assistant.py
# Fournit une encapsulation simple pour construire un résumé et interroger une API LLM

import json


def build_summary_for_api(results_df, task_type='regression', top_n=5):
    """Construit un texte résumant les métriques principales pour l'API LLM.
    Retourne une string concise (tables en CSV si nécessaire).
    """
    # cleaner: ne pas envoyer de confusion matrix brute
    df = results_df.copy()
    cols = [c for c in df.columns if c not in ['confusion_matrix', 'model']]
    df = df[cols]
    # convert to csv (petit tableau) pour faciliter parsing
    csv = df.head(top_n).to_csv(index=False)
    prompt = (
        f"Task type: {task_type}\n"
        f"Top {top_n} models metrics (CSV):\n{csv}\n"
        "Please provide: 1) short justification of the best model, 2) potential next steps for tuning or features."
    )
    return prompt


def ask_model_choice(prompt, api_client=None):
    """Fonction placeholder: en production on envoie prompt via OpenAI/Mistral client.
    Ici on retourne une phrase d'exemple pour garder le pipeline testable sans clef API.
    Si api_client fourni, appeler api_client.create(...) et retourner la réponse.
    """
    if api_client is None:
        return "LLM not configured: example response -> Recommandation: XGBoost for tabular data due to tradeoff performance/time."
    else:
        # Exemple pour OpenAI pseudo-code
        # response = api_client.chat.completions.create(...)
        # return response['choices'][0]['message']['content']
        raise NotImplementedError("Connecter votre client LLM ici")


def combine_api_and_rules(results_df, api_answer, task_type='regression'):
    """Combine la recommandation textuelle de l'API et règles locales pour produire un verdict final."""
    best_by_rules = select_best_model(results_df, task_type=task_type)
    return {
        'api_answer': api_answer,
        'best_by_rules': best_by_rules
    }