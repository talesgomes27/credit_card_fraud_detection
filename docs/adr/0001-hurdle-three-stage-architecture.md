# ADR 0001: Arquitetura em Três Estágios Desacoplados (Hurdle Model)

- **Status:** Aceito
- **Data:** 2026-10-14
- **Contexto:** O dataset apresenta três variáveis target com características estatísticas e objetivos de negócio distintos: detecção binária (`is_fraud`), subtipo de ataque (`fraud_type`) e prejuízo em dólares (`fraud_amount_lost`). As variáveis compartilham forte dependência determinística: transações legítimas possuem invariavelmente perda de \$0,00 e tipo `Legitimate`. A prevalência de fraude é de apenas 2,05%, gerando inflação extrema de zeros na regressão e assimetria severa na classificação.

## Decisão
Adotar uma arquitetura desacoplada em três estágios (*Hurdle Model*):
1. **Estágio 1 (Gatekeeper Binário):** Modelo supervisionado treinado para estimar $P(\text{is\_fraud} = 1 \mid X)$, avaliado por PR-AUC e calibrado financeiramente.
2. **Estágio 2 (Tipificador Multiclasse):** Classificador treinado para discriminar entre as 4 modalidades de fraude (*Card-Not-Present*, *Account-Takeover*, *Synthetic-Identity*, *Card-Present*) avaliado por Macro F1-Score.
3. **Estágio 3 (Quantificador Condicional de Dano):** Regressor treinado estritamente sobre a partição de transações fraudulentas para estimar $\mathbb{E}[\text{fraud\_amount\_lost} \mid \text{is\_fraud} = 1, X]$, avaliado por RMSE e MAE.

A expectativa matemática incondicional de perda por transação é calculada por:
$$\mathbb{E}[\text{Perda} \mid X] = P(\text{is\_fraud}=1 \mid X) \cdot \mathbb{E}[\text{fraud\_amount\_lost} \mid \text{is\_fraud}=1, X]$$

## Alternativas Rejeitadas
- **Classificador Único Multiclasse de 5 Classes:** Agrupar `Legitimate` junto com os 4 subtipos em um único classificador degrada o poder preditivo nas fraudes minoritárias, pois o balanceamento penalizaria o volume maciço de operações legítimas.
- **Regressão Direta Zero-Inflated em Estágio Único:** Regressores lineares ou árvores treinadas diretamente na base completa sofrem com a concentração de 97,95% de zeros, subestimando as perdas de valores elevados na cauda longa.
- **Rede Neural Multi-Task Única:** Dificulta a calibração isolada de thresholds de decisão e inviabiliza o cálculo analítico exato de SHAP via TreeExplainer.

## Consequências
- **Positivas:** Especialização funcional de cada modelo, eliminação do viés de inflação de zeros na regressão, explicabilidade direta por árvore e facilidade de manutenção em produção.
- **Mitigadas:** Erros cometidos no Estágio 1 propagam-se para os estágios subsequentes. Para mitigar esse risco, o limiar de decisão $t^*$ do Estágio 1 é otimizado priorizando recall na curva de custo financeiro.
