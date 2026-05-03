import joblib
import pandas as pd


def load_model(model_path: str):
    """
    	Carregar pipeline de classificação de obesidade treinado.
    """
    return joblib.load(model_path)


def predict_obesity(model, input_data: dict):
    """
	Prever o grau de obesidade a partir de dados brutos do paciente.
	O modelo não utiliza altura, peso ou IMC.
    """
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]

    return prediction


def predict_obesity_with_proba(model, input_data: dict):
    """
    	Prever o grau de obesidade e as probabilidades de retorno, quando disponíveis.
    """
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_df)[0]
        classes = model.classes_

        proba_df = pd.DataFrame({
            "class": classes,
            "probability": probabilities
        }).sort_values(by="probability", ascending=False)

        return prediction, proba_df

    return prediction, None