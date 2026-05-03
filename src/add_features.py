import pandas as pd

def add_features(data):
    
    df_feature = data.copy()

    # Faixa etária
    df_feature["faixa_etaria"] = pd.cut(
        df_feature["idade"],
        bins=[0, 18, 30, 45, 60, 120],
        labels=["Abaixo 18", "18-30", "31-45", "46-60", "60+"],
        include_lowest=True
    )

    # Transporte ativo
    df_feature["transporte_ativo"] = df_feature["meio_transporte"].isin(["Walking", "Bike"]).astype(int)

    # Indicadores binários de estilo de vida
    risco_alimento_cal = df_feature["freq_alimentos_caloricos"].map({"no": 0, "yes": 1}).fillna(0)
    risco_familia = df_feature["historico_familia_sobrepeso"].map({"no": 0, "yes": 1}).fillna(0)
    monitora_caloria = df_feature["monitora_calorias_dia"].map({"no": 0, "yes": 1}).fillna(0)

    # NVariáveis comportamentais numéricas
    freq_exercicio = 3 - df_feature["freq_exercicios"]
    uso_tecnologia = df_feature["tempo_dispositivos_eletronicos"]
    
    # Mapeamento ordinal 
    consumo_map = {
        "no": 0,
        "Sometimes": 1,
        "Frequently": 2,
        "Always": 3
    }
    risco_consumo = df_feature["consumo_lanches_entre_refeicoes"].map(consumo_map).fillna(0)

    # Lifestyle score
    df_feature["score_lifestyle"] = (
        risco_familia
        + risco_alimento_cal
        + risco_consumo
        + freq_exercicio
        + uso_tecnologia
        - monitora_caloria
    )

    return df_feature