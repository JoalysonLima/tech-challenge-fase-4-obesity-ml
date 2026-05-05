# ============================================================
# Aplicação Streamlit - Sistema Preditivo de Obesidade
# Tech Challenge - Fase 4
#
# Objetivo:
# - Carregar o modelo treinado.
# - Permitir que a equipe médica informe dados do paciente.
# - Retornar a classe prevista de obesidade.
# - Apresentar um painel analítico com insights do estudo.
# - Mostrar métricas de avaliação do modelo.
# ============================================================

# ============================================================
# Importação das bibliotecas
# ============================================================
import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

# Importando a função de feature engineering
import sys
from pathlib import Path

# Caminho da pasta app/
APP_DIR = Path(__file__).resolve().parent

# Caminho da raiz do projeto
PROJECT_ROOT = APP_DIR.parent

# Adiciona a raiz do projeto ao caminho de importação do Python
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Importa a função usada dentro do pipeline salvo
from src.add_features import add_features

# ============================================================
# Configuração inicial da página
# ============================================================
st.set_page_config(
    page_title="Sistema de Apoio à Classificação de Obesidade",
    page_icon="🏥",
    layout="wide"
)

# ============================================================
# Definição dos caminhos do projeto
# BASE_DIR aponta para a pasta app/
# ============================================================
APP_DIR = Path(__file__).resolve().parent

# PROJECT_DIR aponta para a raiz do projeto
PROJECT_DIR = APP_DIR.parent

# Caminhos dos arquivos usados pela aplicação
MODEL_PATH = PROJECT_DIR / "models" / "obesity_model_pipeline.joblib"
METADATA_PATH = PROJECT_DIR / "models" / "model_metadata.json"
METRICS_PATH = PROJECT_DIR / "reports" / "metrics" / "final_model_metrics.json"
DATA_PATH = PROJECT_DIR / "data" / "raw" / "Obesity.csv"


# ============================================================
# Funções de carregamento
# ============================================================
@st.cache_resource
def carregar_modelo():
    # Carrega o pipeline treinado.
    return joblib.load(MODEL_PATH)


@st.cache_data
def carregar_dados():
    """
    Carrega a base original utilizada no projeto.
    A remoção de duplicados é feita aqui para manter coerência com
    a etapa de diagnóstico/modelagem.
    """
    df = pd.read_csv(DATA_PATH)
    df = df.drop_duplicates().copy()
    return df


@st.cache_data
def carregar_json(caminho):
    """
    Carrega arquivos JSON, como métricas e metadados do modelo.
    """
    if caminho.exists():
        with open(caminho, "r", encoding="utf-8") as arquivo:
            return json.load(arquivo)

    return {}


# ============================================================
# Funções auxiliares de predição e explicação
# ============================================================

def prever_obesidade(modelo, dados_entrada):
    """
    Recebe os dados preenchidos no formulário,
    converte para DataFrame e retorna a predição do modelo.

    Parâmetros
    -----------------------------------------------
    modelo:
        Pipeline treinado do scikit-learn.

    dados_entrada:
        Dicionário com os dados do paciente.

    Retorno
    -----------------------------------------------
    predicao:
        Classe prevista de obesidade.

    probabilidades_df:
        DataFrame com probabilidades por classe utilizando predict_proba.
    """
    entrada_df = pd.DataFrame([dados_entrada])

    predicao = modelo.predict(entrada_df)[0]

    # hasattr() verifica se o objeto modelo possui um método/atributo chamado predict_proba
    if hasattr(modelo, "predict_proba"):
        probabilidades = modelo.predict_proba(entrada_df)[0]
        classes = modelo.classes_

        probabilidades_df = pd.DataFrame({
            "Classe": classes,
            "Probabilidade": probabilidades
        }).sort_values(by="Probabilidade", ascending=False)

        return predicao, probabilidades_df

    return predicao, None


def calcular_indicador_comportamental(dados_entrada):
    """
    Calcula uma prévia do indicador comportamental usado para explicação no app.
    """
    risco_alimento_calorico = 1 if dados_entrada["freq_alimentos_caloricos"] == "yes" else 0
    risco_historico_familiar = 1 if dados_entrada["historico_familia_sobrepeso"] == "yes" else 0
    protecao_monitoramento_calorico = 1 if dados_entrada["monitora_calorias_dia"] == "yes" else 0

    mapa_lanches = {
        "no": 0,
        "Sometimes": 1,
        "Frequently": 2,
        "Always": 3
    }

    risco_lanches = mapa_lanches.get(dados_entrada["consumo_lanches_entre_refeicoes"], 0)

    # Quanto menor a atividade física, maior o valor de inatividade.
    inatividade_fisica = 3 - dados_entrada["freq_exercicios"]

    # tempo_dispositivos_eletronicos representa tempo de uso de dispositivos tecnológicos.
    uso_tecnologia = dados_entrada["tempo_dispositivos_eletronicos"]

    indicador = (
        risco_historico_familiar
        + risco_alimento_calorico
        + risco_lanches
        + inatividade_fisica
        + uso_tecnologia
        - protecao_monitoramento_calorico
    )

    return round(indicador, 2)


def traduzir_classe_obesidade(classe):
    """
    Traduz as classes originais do dataset para uma descrição em português.
    """
    traducoes = {
        "Insufficient_Weight": "Peso insuficiente",
        "Normal_Weight": "Peso normal",
        "Overweight_Level_I": "Sobrepeso nível I",
        "Overweight_Level_II": "Sobrepeso nível II",
        "Obesity_Type_I": "Obesidade tipo I",
        "Obesity_Type_II": "Obesidade tipo II",
        "Obesity_Type_III": "Obesidade tipo III"
    }

    return traducoes.get(classe, classe)


# ============================================================
# Carregamento dos arquivos principais
# ============================================================

try:
    modelo = carregar_modelo()
except Exception as erro:
    st.error(
        "Não foi possível carregar o modelo. "
        "Verifique se o arquivo obesity_model_pipeline.joblib está na pasta models/."
    )
    st.exception(erro)
    st.stop()

try:
    df = carregar_dados()
except Exception as erro:
    st.error(
        "Não foi possível carregar a base de dados. "
        "Verifique se o arquivo Obesity.csv está na pasta data/raw/."
    )
    st.exception(erro)
    st.stop()

metadados = carregar_json(METADATA_PATH)
metricas = carregar_json(METRICS_PATH)


# ============================================================
# Barra lateral de navegação
# ============================================================

st.sidebar.title("Navegação")

pagina = st.sidebar.radio(
    "Selecione uma página",
    [
        "Visão Geral",
        "Sistema Preditivo",
        "Performance do Modelo",
        "Notas do Projeto"
    ]
)

st.sidebar.markdown("---")

# ============================================================
# Página 1 - Visão Geral
# ============================================================

if pagina == "Visão Geral":
    st.title("Sistema de Apoio à Classificação de Obesidade")
    st.subheader("Ferramenta preditiva para apoio à tomada de decisão em ambiente hospitalar")

    st.markdown(
        """
        Este projeto foi desenvolvido no contexto de um hospital, com o objetivo de apoiar
        médicos e médicas na estimativa do possível nível de obesidade de uma pessoa a partir
        de informações demográficas, histórico familiar e hábitos de vida.

        A solução combina duas entregas principais:

        1. **Sistema preditivo**: permite inserir dados de um paciente e obter uma classe prevista.
        2. **Dashboard analítico**: apresenta padrões observados na base de dados sobre obesidade.

        O objetivo não é substituir a avaliação clínica, mas apoiar triagem, análise preventiva
        e discussões baseadas em dados.
        """
    )

    st.warning(
        "Importante: esta aplicação não realiza diagnóstico médico. "
        "Ela deve ser usada apenas como ferramenta de apoio à decisão."
    )

    st.markdown("### Variável alvo")

    st.code("Obesity", language="text")

    st.markdown("### Classes previstas pelo modelo")

    classes = metadados.get(
        "classes",
        sorted(df["Obesity"].unique().tolist())
    )

    for classe in classes:
        st.markdown(f"- `{classe}` — {traduzir_classe_obesidade(classe)}")

    st.markdown("### Decisão metodológica principal")

    st.markdown(
        """
        Nesta versão do projeto, as variáveis `altura`, `peso` e `IMC` foram excluídas
        do modelo preditivo.

        Essa decisão foi tomada porque os níveis de obesidade estão fortemente relacionados
        a medidas corporais. Se altura, peso ou IMC fossem utilizados, o modelo poderia
        apenas reproduzir uma regra próxima à classificação corporal tradicional.

        Com essa abordagem, o modelo passa a avaliar o potencial preditivo de variáveis
        relacionadas a comportamento, hábitos de vida, histórico familiar e perfil demográfico.
        """
    )


# ============================================================
# Página 2 - Sistema Preditivo
# ============================================================

elif pagina == "Sistema Preditivo":
    st.title("Sistema Preditivo")
    st.subheader("Estimativa do nível de obesidade com base no perfil do paciente")

    st.markdown(
        """
        Preencha os dados abaixo para gerar uma predição.
        """
    )

    with st.form("formulario_predicao"):
        coluna_1, coluna_2, coluna_3 = st.columns(3)

        with coluna_1:
            genero = st.selectbox(
                "Gênero",
                ["Female", "Male"],
                format_func=lambda x: "Feminino" if x == "Female" else "Masculino"
            )

            idade = st.slider(
                "Idade",
                min_value=14.0,
                max_value=61.0,
                value=25.0,
                step=1.0
            )

            historico_familia_sobrepeso = st.selectbox(
                "Histórico familiar de excesso de peso",
                ["yes", "no"],
                format_func=lambda x: "Sim" if x == "yes" else "Não"
            )

            freq_alimentos_caloricos = st.selectbox(
                "Consome alimentos altamente calóricos com frequência?",
                ["yes", "no"],
                format_func=lambda x: "Sim" if x == "yes" else "Não"
            )

            fumante = st.selectbox(
                "Fuma?",
                ["no", "yes"],
                format_func=lambda x: "Sim" if x == "yes" else "Não"
            )

        with coluna_2:
            freq_consumo_vegetais = st.slider(
                "Frequência de consumo de vegetais nas refeições",
                min_value=1.0,
                max_value=3.0,
                value=2.0,
                step=0.1,
                help="Escala aproximada: 1 = raramente, 2 = às vezes, 3 = sempre."
            )

            num_refeicoes_diarias = st.slider(
                "Número de refeições principais por dia",
                min_value=1.0,
                max_value=4.0,
                value=3.0,
                step=0.1,
                help="Escala aproximada: 1 = uma refeição, 2 = duas, 3 = três, 4 = quatro ou mais."
            )

            consumo_lanches_entre_refeicoes = st.selectbox(
                "Come algo entre as refeições?",
                ["no", "Sometimes", "Frequently", "Always"],
                format_func=lambda x: {
                    "no": "Não",
                    "Sometimes": "Às vezes",
                    "Frequently": "Frequentemente",
                    "Always": "Sempre"
                }.get(x, x)
            )

            consumo_diario_agua = st.slider(
                "Consumo diário de água",
                min_value=1.0,
                max_value=3.0,
                value=2.0,
                step=0.1,
                help="Escala aproximada: 1 = menos de 1L, 2 = entre 1L e 2L, 3 = mais de 2L."
            )

        with coluna_3:
            monitora_calorias_dia = st.selectbox(
                "Monitora o consumo diário de calorias?",
                ["no", "yes"],
                format_func=lambda x: "Sim" if x == "yes" else "Não"
            )

            freq_exercicios = st.slider(
                "Frequência de atividade física",
                min_value=0.0,
                max_value=3.0,
                value=1.0,
                step=0.1,
                help="Escala aproximada: 0 = nenhuma, 1 = 1-2x/semana, 2 = 3-4x/semana, 3 = 5x ou mais."
            )

            tempo_dispositivos_eletronicos = st.slider(
                "Tempo diário usando dispositivos tecnológicos",
                min_value=0.0,
                max_value=2.0,
                value=1.0,
                step=0.1,
                help="Escala aproximada: 0 = 0-2h/dia, 1 = 3-5h/dia, 2 = mais de 5h/dia."
            )

            consumo_alcool = st.selectbox(
                "Consumo de álcool",
                ["no", "Sometimes", "Frequently", "Always"],
                format_func=lambda x: {
                    "no": "Não consome",
                    "Sometimes": "Às vezes",
                    "Frequently": "Frequentemente",
                    "Always": "Sempre"
                }.get(x, x)
            )

            meio_transporte = st.selectbox(
                "Meio de transporte habitual",
                [
                    "Automobile",
                    "Motorbike",
                    "Bike",
                    "Public_Transportation",
                    "Walking"
                ],
                format_func=lambda x: {
                    "Automobile": "Carro",
                    "Motorbike": "Moto",
                    "Bike": "Bicicleta",
                    "Public_Transportation": "Transporte público",
                    "Walking": "A pé"
                }.get(x, x)
            )

        botao_predicao = st.form_submit_button("Gerar predição")

    if botao_predicao:
        dados_entrada = {
            "idade": idade,
            "genero": genero,
            "historico_familia_sobrepeso": historico_familia_sobrepeso,
            "freq_alimentos_caloricos": freq_alimentos_caloricos,
            "fumante": fumante,
            "freq_consumo_vegetais": freq_consumo_vegetais,
            "num_refeicoes_diarias": num_refeicoes_diarias,
            "consumo_lanches_entre_refeicoes": consumo_lanches_entre_refeicoes,
            "consumo_diario_agua": consumo_diario_agua,
            "monitora_calorias_dia": monitora_calorias_dia,
            "freq_exercicios": freq_exercicios,
            "tempo_dispositivos_eletronicos": tempo_dispositivos_eletronicos,
            "consumo_alcool": consumo_alcool,
            "meio_transporte": meio_transporte
        }

        predicao, probabilidades_df = prever_obesidade(modelo, dados_entrada)
        indicador_comportamental = calcular_indicador_comportamental(dados_entrada)

        st.markdown("---")
        st.subheader("Resultado da predição")

        st.success(
            f"Classe prevista: **{traduzir_classe_obesidade(predicao)}** (`{predicao}`)"
        )

        st.metric(
            label="Indicador comportamental exploratório",
            value=indicador_comportamental
        )

        st.info(
            "O indicador comportamental é apenas uma medida exploratória baseada nas variáveis "
            "do dataset. Ele não é um score clínico validado."
        )

        if probabilidades_df is not None:
            st.markdown("### Probabilidades por classe")

            probabilidades_df["Probabilidade (%)"] = probabilidades_df["Probabilidade"] * 100
            probabilidades_df["Classe traduzida"] = probabilidades_df["Classe"].apply(traduzir_classe_obesidade)

            fig = px.bar(
                probabilidades_df,
                x="Probabilidade (%)",
                y="Classe traduzida",
                orientation="h",
                title="Probabilidade estimada por classe",
                text=probabilidades_df["Probabilidade (%)"].round(1)
            )

            fig.update_layout(
                yaxis={"categoryorder": "total ascending"},
                xaxis_title="Probabilidade (%)",
                yaxis_title="Classe"
            )

            st.plotly_chart(fig, use_container_width=True)

        st.warning(
            "Esta predição deve ser usada apenas como apoio à decisão. "
            "A avaliação final deve ser feita por profissional de saúde."
        )

        resultado_download = {
        "nível_previsto": predicao,
        "probabilidade_estimada": round(float(probabilidades_df["Probabilidade"].head(1)), 2),
        "observacao": "Resultado apenas para apoio à decisão. Não substitui avaliação médica."
        }

        st.download_button(
            label="Baixar resultado em JSON",
            data=json.dumps(resultado_download, indent=4, ensure_ascii=False),
            file_name="resultado_predicao_obesidade.json",
            mime="application/json"
        )


# ============================================================
# Página 3 - Performance do Modelo
# ============================================================

elif pagina == "Performance do Modelo":
    st.title("Performance do Modelo")
    st.subheader("Avaliação do modelo final de classificação multiclasse")

    if metricas:
        metrica_1, metrica_2, metrica_3, metrica_4 = st.columns(4)

        with metrica_1:
            st.metric("Accuracy", metricas.get("accuracy", "N/A"))

        with metrica_2:
            st.metric("F1 Macro", metricas.get("f1_macro", "N/A"))

        with metrica_3:
            st.metric("Precision Macro", metricas.get("precision_macro", "N/A"))

        with metrica_4:
            st.metric("Recall Macro", metricas.get("recall_macro", "N/A"))

        st.markdown("### Metadados do modelo")

        st.json(metadados)

    else:
        st.warning(
            "Arquivo de métricas não encontrado. "
            "Verifique se final_model_metrics.json está em reports/metrics/."
        )

    st.markdown("### Interpretação da avaliação")

    st.markdown(
        """
        Como o problema é de classificação multiclasse em um contexto de saúde,
        a avaliação não deve depender apenas da accuracy. Também devem ser analisados:

        - precision;
        - recall;
        - F1-score;
        - matriz de confusão;
        - desempenho por classe.
        """
    )

# ============================================================
# Página 4 - Notas do Projeto
# ============================================================

elif pagina == "Notas do Projeto":
    st.title("Notas do Projeto")
    st.subheader("Metodologia, limitações e uso responsável")

    st.markdown(
        """
        ## Metodologia

        O projeto segue uma pipeline completa de Machine Learning:

        1. Diagnóstico dos dados.
        2. Análise exploratória.
        3. Feature engineering.
        4. Pré-processamento com pipeline.
        5. Treinamento de modelos.
        6. Avaliação com métricas multiclasse.
        7. Deploy em Streamlit.
        8. Construção de dashboard analítico.

        ## Decisão metodológica principal

        O modelo final foi construído sem as variáveis:

        - `altura`;
        - `peso`;
        - `IMC`.

        Essa escolha busca reduzir a dependência de medidas corporais e avaliar
        o quanto variáveis de hábitos de vida e perfil demográfico conseguem contribuir
        para a classificação dos níveis de obesidade.

        ## Limitações

        - O modelo não realiza diagnóstico médico.
        - O modelo não prova causalidade.
        - A base de dados possui tamanho limitado.
        - As predições dependem da qualidade dos dados informados.

        ## Uso responsável

        O sistema deve ser usado como ferramenta de apoio à decisão e triagem.
        A decisão clínica final deve permanecer sob responsabilidade de profissionais
        de saúde qualificados.
        """
    )