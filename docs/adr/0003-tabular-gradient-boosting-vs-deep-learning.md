# ADR 0003: Seleção de Algoritmos de Gradient Boosting sobre Redes Neurais Profundas

- **Status:** Aceito
- **Data:** 2026-10-14
- **Contexto:** A modelagem preditiva tabular envolve 40 features heterogêneas (contagens inteiras, proporções contínuas, coordenadas de distância e categorias de alta/média cardinalidade). O volume de dados é de 100.000 linhas, e o sistema exige interpretabilidade rigorosa para justificar bloqueios de cartões perante auditorias de conformidade e clientes.

## Decisão
Selecionar algoritmos baseados em árvores de decisão impulsionadas por gradiente (**LightGBM** e **XGBoost**) como a família principal de modelos para os Estágios 1, 2 e 3, mantendo **Regressão Logística Ponderada** e **Random Forest** como baselines interpretáveis.

## Alternativas Rejeitadas
- **Deep Learning Tabular (FT-Transformer, TabNet, MLP Profundo):** Exigem volume de parâmetros desproporcional, pré-processamento minucioso de normalização/embeddings, elevado custo computacional de treinamento e são propensos a instabilidades em dados com assimetria extrema de 2% de positivos. Além disso, a interpretabilidade via Integrated Gradients é ordens de magnitude mais lenta que o cálculo exato de SHAP em árvores.

## Consequências
- **Positivas:**
  1. Suporte nativo a variáveis categóricas e tratamento natural de valores numéricos sem necessidade de transformações estritas de normalidade.
  2. Implementação eficiente de ponderação de amostras e funções de perda customizadas (`scale_pos_weight`).
  3. Cálculo determinístico e rápido de valores de Shapley através do algoritmo analítico `TreeExplainer` da biblioteca `shap`, com complexidade $O(TLD^2)$.
  4. Tempos de treinamento na ordem de segundos, permitindo varreduras consistentes de hiperparâmetros e validações rápidas.
- **Mitigadas:** Modelos de árvores não extrapolam além dos limites numéricos observados no treino para a regressão do Estágio 3. Isso é mitigado pelo fato de que as transações fraudulentas têm limites físicos e financeiros superiores estabelecidos por contratos e limites de crédito.
