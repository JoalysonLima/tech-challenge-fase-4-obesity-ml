# Riscos, hipóteses e limitações

## 1. Riscos metodológicos

### 1.1 Data leakage

Existe risco de vazamento de dados caso transformações como imputação, normalização, encoding, seleção de features ou balanceamento de classes sejam ajustadas antes da separação entre treino e teste.

Tratamento: utilizar `Pipeline`, `ColumnTransformer` e, quando houver técnicas de oversampling, `Pipeline` do `imblearn`, garantindo que os ajustes sejam aprendidos apenas com os dados de treino dentro de cada etapa de validação.

### 1.2 Overfitting

Modelos mais complexos, como pode ser o caso, podem apresentar desempenho artificialmente alto no conjunto de treino e pior desempenho em dados novos. Técnicas de oversampling, quando mal aplicadas, também podem aumentar esse risco ao duplicar ou criar exemplos sintéticos semelhantes aos dados já existentes.

Tratamento: usar validação adequada, comparação entre modelos, controle de hiperparâmetros e análise final de desempenho no conjunto de teste.

### 1.3 Desbalanceamento de classes

Como a variável alvo possui múltiplas categorias de estado nutricional, pode haver diferença relevante na quantidade de observações entre as classes. Nesse cenário, a métrica de accuracy pode ser enganosa, pois o modelo pode apresentar bom desempenho geral enquanto classifica mal classes menos frequentes.

Tratamento: além da accuracy, serão avaliadas métricas mais adequadas para classificação multiclasse desbalanceada, como balanced accuracy, macro F1-score, recall por classe e matriz de confusão.

Estratégia metodológica: o balanceamento das classes não será aplicado antes da separação entre treino e teste, para evitar data leakage. Caso o desbalanceamento afete o desempenho das classes minoritárias, serão comparadas diferentes abordagens durante a etapa de modelagem, como:

- modelo sem balanceamento;
- modelos com `class_weight="balanced"`;
- Random Oversampling;
- SMOTE ou SMOTENC, quando for tecnicamente adequado.

Como o dataset contém variáveis categóricas e numéricas, o uso de SMOTE simples exige cautela. Caso seja utilizada uma técnica sintética de oversampling, será dada preferência ao SMOTENC ou a uma estratégia mais compatível com dados variados.

### 1.4 Uso em contexto médico

O modelo não deve substituir avaliação médica. A solução tem finalidade educacional e analítica, não sendo adequada para diagnóstico clínico real sem validação médica. Ele deve ser interpretado como sistema de apoio à decisão, oferecendo uma estimativa preditiva baseada nos dados disponíveis.

## 2. Inconsistências da descrição do challenge

### 2.1 Termo “assertividade”

A descrição exige assertividade acima de 75%, mas não define explicitamente qual métrica representa essa assertividade.

Decisão do projeto: interpretar assertividade como accuracy, por ser a métrica mais diretamente associada à proporção total de acertos. No entanto, a avaliação será complementada com métricas mais robustas para classificação multiclasse, como balanced accuracy, macro F1-score, recall por classe e matriz de confusão.

### 2.2 Previsão de obesidade versus categoria de obesidade

A descrição menciona prever se uma pessoa “pode ter obesidade”, sugerindo uma formulação binária. No entanto, a base possui múltiplas categorias relacionadas ao estado nutricional.

Decisão do projeto: tratar o problema como classificação multiclasse, preservando a granularidade da variável alvo original. Essa escolha permite diferenciar níveis como peso normal, sobrepeso e diferentes tipos de obesidade, oferecendo uma análise mais informativa para apoio à decisão.