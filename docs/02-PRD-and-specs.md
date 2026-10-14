---
title: "Documento de Requisitos de Produto (PRD) — Gate 2"
subtitle: "Especificação de negócio, métricas assimétricas e matriz 1:1 de rastreabilidade"
author: "Tales"
date: "2026-10-14"
---

- **Projeto:** Detecção, Tipificação e Quantificação de Fraudes em Cartão de Crédito
- **Arquétipo:** A — Data Science, Analytics & Machine Learning
- **Artefato:** `docs/02-PRD-and-specs.md`

---

## 1. Visão Geral do Problema e Contexto de Negócio

Fraudes financeiras com cartões de pagamento causam prejuízos bilionários anuais a emissores, adquirentes e lojistas.
Além do valor direto subtraído na transação, as instituições arcam com custos administrativos de estorno (*chargebacks*), despesas de análise operacional e atrito com clientes legítimos bloqueados por falsos positivos.

O dataset inspecionado no Gate 1 disponibiliza 100.000 transações de 2026.
Os atributos refletem comportamentos reais do sistema financeiro, sem anonimização artificial por PCA.
O objetivo central deste projeto é desenhar, validar e documentar um pipeline analítico em múltiplos estágios (*Hurdle Model*) capaz de:
1. Detectar transações fraudulentas em tempo real com alta precisão e sensibilidade calibrada.
2. Identificar a modalidade específica da fraude entre as 4 classes ativas do ecossistema.
3. Estimar o dano financeiro residual esperado pós-recuperação parcial para priorização de filas de contestação.
4. Comunicar os achados técnicos e recomendações operacionais por meio de um relatório executivo em Quarto publicado no GitHub Pages.

### Stakeholders e Usuários Primários
- **Equipe de Risco e Prevenção à Fraude:** Demanda regras claras, pontuação de probabilidade contínua e interpretabilidade de decisões (SHAP) para justificar bloqueios a clientes.
- **Operações de Chargeback:** Necessita de triagem por subtipo de fraude e valor estimado para direcionar esforços de recuperação de crédito.
- **Liderança Executiva (C-Level / Head de Risco):** Requer visibilidade do balanço econômico entre perda financeira evitada e custo de fricção operacional.

---

## 2. Questões Norteadoras da Fase ASK e Hipóteses de Negócio

### 2.1 Questões Norteadoras Centrais
- **Q1:** Qual o padrão temporal das fraudes ao longo das 24 horas e qual o incremento de risco na madrugada?
- **Q2:** Como fatores geográficos (operações internacionais e jurisdições de alto risco) amplificam a taxa de fraude?
- **Q3:** Qual a redução empírica de risco proporcionada por autenticações fortes (Chip EMV e senha PIN) em relação a transações não presenciais?
- **Q4:** Em que medida o comportamento recente de velocidade de transações e a distorção do ticket médio sinalizam ataques em andamento?
- **Q5:** Qual o ponto de corte ótimo de probabilidade para minimizar a perda financeira global considerando o custo de falso positivo?
- **Q6:** Quais atributos do perfil do cliente e da transação mais influenciam o algoritmo em nível global e local?
- **Q7:** O modelo multiclasse consegue separar satisfatoriamente ataques cibernéticos (*Account Takeover*) de fraudes de identidade sintética?
- **Q8:** A estimativa de perda financeira residual apresenta resíduos homocedásticos e erro médio compatível com os limites operacionais?

### 2.2 Hipóteses de Negócio e Engenharia (H1 a H8)

- **H1 (Janela Noturna):** Transações efetuadas entre 01h e 04h apresentam odds ratio superior a 3,5x em comparação com o horário comercial regular, devido à menor vigilância ativa do titular.
- **H2 (Geografia de Risco):** Estabelecimentos sediados no quarteto de risco (China, Nigéria, Romênia e Rússia) concentram taxas de fraude superiores a 10%, funcionando como red flags independentes.
- **H3 (Barreira Física EMV):** A presença simultânea de leitura de chip e validação de PIN reduz a probabilidade de fraude para menos de 0,6%, tornando transações presenciais seguras contra clonagem direta.
- **H4 (Disparada de Frequência):** Três ou mais transações no intervalo de uma hora (`transactions_last_1h >= 3`) indicam testes repetidos de cartão, multiplicando a taxa de fraude em pelo menos 2,5x.
- **H5 (Desvio de Ticket Médio):** Compras com razão `amount_vs_avg_ratio >= 5.0` concentram perdas financeiras desproporcionais, justificando regras de bloqueio preventivo condicional.
- **H6 (Assimetria de Custos):** O threshold padrão de 0,50 é subótimo para o negócio. Ajustar o corte para balancear custo de falso positivo (\$3,00) contra perda média (\$136,18) reduz o custo operacional global.
- **H7 (Separabilidade por Subtipo):** Fraudes de *Card-Not-Present* são primariamente guiadas por canal digital e ausência de chip, enquanto *Account Takeover* é caracterizada por salto geográfico (`distance_from_last_tx_km`) e hiato temporal prévio prolongado.
- **H8 (Condicionamento de Perda):** Modelar o montante residual apenas sobre a partição fraudulenta (*Hurdle regression*) gera resíduos com menor viés em comparação a uma regressão linear direta sobre a base zero-inflada.

---

## 3. Justificativa Matemática das Métricas de Avaliação

A escolha das métricas reflete a física do problema em cada estágio do pipeline:

```
                  ┌─────────────────────────────────────────────────────────┐
                  │                 Dataset Bruto (100k)                    │
                  └───────────────────────────┬─────────────────────────────┘
                                              │
                                              ▼
                    ┌─────────────────────────────────────────────────────┐
                    │      Estágio 1: is_fraud (Classificação Binária)     │
                    │      Prevalência: 2,05%  |  Métrica: PR-AUC & Custo │
                    └─────────────────────────┬───────────────────────────┘
                                              │
                            ┌─────────────────┴─────────────────┐
                            │ (Predição Positiva: Fraude)       │
                            ▼                                   ▼
          ┌───────────────────────────────────┐ ┌───────────────────────────────────┐
          │   Estágio 2: fraud_type           │ │   Estágio 3: fraud_amount_lost    │
          │   Classificação Multiclasse (4)   │ │   Regressão Condicional ($)       │
          │   Métrica: Macro F1-Score         │ │   Métrica: RMSE e MAE             │
          └───────────────────────────────────┘ └───────────────────────────────────┘
```

### 3.1 Estágio 1 — Classificação Binária (`is_fraud`)

- **Métrica Primária: PR-AUC (Área sob a Curva Precision-Recall)**
  - *Justificativa Matemática:* A base possui apenas 2,05% de eventos positivos ($N_{positivo} = 2.055$). Em bases severamente desbalanceadas, a curva ROC-AUC é enganosa. Como a taxa de falsos positivos ($FPR = \frac{FP}{FP + TN}$) tem no denominador o universo maciço de legítimas ($TN \approx 98.000$), mesmo centenas de falsos alarmes mantêm o $FPR$ ínfimo, inflando artificialmente o ROC-AUC para acima de 0,92.
  - A curva Precision-Recall exclui $TN$ de sua formulação. Avalia diretamente o compromisso entre Precisão ($\frac{TP}{TP + FP}$) e Recall ($\frac{TP}{TP + FN}$):
    $$\text{PR-AUC} = \int_{0}^{1} P(R) \, dR$$
    O baseline aleatório no PR-AUC é exatamente a prevalência histórica ($\approx 0,0205$), permitindo medir ganho real sobre o chute aleatório.
- **Métrica de Negócio: Função de Custo Financeiro Total ($C_{total}$)**
  - Seja $t \in [0, 1]$ o limiar de decisão. Para cada transação $i$:
    $$C(t) = \sum_{i \in FN(t)} \text{PerdaReal}_i + \sum_{i \in FP(t)} C_{atrito}$$
    Assumimos como premissa conservadora de mercado que cada falso positivo gera um custo operacional e de atrito de $C_{atrito} = \$3,50$ (contato via SMS/push e suporte), enquanto o falso negativo custa o valor integral não recuperado da transação fraudulenta. O ponto ótimo $t^*$ é obtido por:
    $$t^* = \arg\min_{t} C(t)$$
- **Métricas Secundárias:** Recall na faixa de alta precisão (ex: Precision $\ge 0,80$), F1-Score no limiar ótimo e ROC-AUC para comparação bibliográfica.

### 3.2 Estágio 2 — Classificação Multiclasse (`fraud_type`)

- **Métrica Primária: Macro F1-Score**
  - *Justificativa Matemática:* A distribuição entre os 4 subtipos de fraude no universo fraudulento é altamente desigual: Card-Not-Present (51,3%), Account-Takeover (21,8%), Synthetic-Identity (15,0%) e Card-Present (11,9%).
  - O cálculo Micro F1 ou Acurácia Simples seria dominado pela categoria majoritária. O Macro F1 calcula a média aritmética não ponderada dos $F_1$ individuais de cada classe $k$:
    $$\text{Macro } F_1 = \frac{1}{K} \sum_{k=1}^{K} \frac{2 \cdot P_k \cdot R_k}{P_k + R_k}$$
    Isso penaliza severamente o modelo caso ele falhe em identificar modalidades raras, como *Card-Present* fraudulento.
- **Métricas Secundárias:** Balanced Accuracy e Matriz de Confusão Normalizada por Linha (Recall por classe).

### 3.3 Estágio 3 — Quantificação Contínua de Dano (`fraud_amount_lost`)

- **Métrica Primária: RMSE (Root Mean Squared Error) em Instâncias Fraudulentas**
  - *Justificativa Matemática:* A distribuição de perda em fraudes varia de \$0,58 a \$4.482,16 (média \$136,18, desvio padrão \$273,26). O RMSE penaliza desproporcionalmente desvios em fraudes de grande magnitude:
    $$\text{RMSE} = \sqrt{\frac{1}{N_{fraude}} \sum_{i=1}^{N_{fraude}} (y_i - \hat{y}_i)^2}$$
    Como o prejuízo institucional se concentra na cauda longa de valores altos, o RMSE é o estimador adequado para calibração de reservas de capital.
- **Métrica Secundária: MAE (Mean Absolute Error)**
  - O MAE quantifica o erro mediano típico da perda em dólares sem a hipersensibilidade a outliers.

---

## 4. Estratégia de Partição e Protocolo de Validação

Para impedir vazamento temporal (*lookahead bias*) e respeitar a dinâmica anual de 2026, adota-se **divisão temporal out-of-time cronológica**:

| Partição | Período Cronológico | Volume de Transações | Proporção | Fraudes Esperadas | Papel Metodológico |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Treino** | Jan a Ago / 2026 (`2026-01-01` a `2026-08-31`) | ~66.860 | ~66,9% | ~1.378 (2,06%) | Treinamento de pesos e calibração interna de árvores. |
| **Validação** | Set a Out / 2026 (`2026-09-01` a `2026-10-31`) | ~16.905 | ~16,9% | ~362 (2,14%) | Tuning de hiperparâmetros, corte ótimo $t^*$ e seleção de modelos. |
| **Teste Cego** | Nov a Dez / 2026 (`2026-11-01` a `2026-12-31`) | ~16.235 | ~16,2% | ~315 (1,94%) | Avaliação final intocada de generalização out-of-time. |

**Regras estritas de validação:**
1. A base inteira será previamente ordenada por `transaction_datetime` crescente para corrigir a inversão pontual detectada em 28/02.
2. Todas as transformações estatísticas (médias de encoding, escaladores numéricos e pipelines) serão ajustadas estritamente com os dados de Treino, aplicando-se apenas o método `.transform()` nos conjuntos de Validação e Teste.
3. Fixação mandatória de semente determinística (`random_state=42`) em qualquer operação amostral.

---

## 5. Matriz 1:1 de Rastreabilidade (Ask ➔ Métrica ➔ Share Quarto)

Conforme as diretrizes do `quarto-quality-checklist.md`, toda pergunta de negócio possui correspondência unívoca com uma métrica quantitativa e uma visualização planejada para o relatório executivo:

| ID | Pergunta de Negócio (Ask) | Métrica / Hipótese | Figura Planejada na Fase Share | Especificações Visuais |
| :--- | :--- | :--- | :--- | :--- |
| **P1** | Como a taxa de fraude varia por hora do dia e onde está o pico de risco? | Taxa de fraude horária (%) e Odds Ratio da madrugada (H1). | `fig01_hourly_fraud_risk` | Painel com barras de volume e linha de taxa de fraude horária, com preenchimento sombreado na faixa de 01h–04h e anotação do multiplicador 4,15x. |
| **P2** | Qual o impacto do país do lojista na concentração de ataques? | Taxa de fraude por país vs. baseline global (H2). | `fig02_country_risk_multiplier` | Gráfico de barras horizontais ordenadas com linha de referência vertical no baseline (2,05%) e realce nas 4 jurisdições críticas. |
| **P3** | Autenticação física (Chip e PIN) anula o risco de fraude? | Redução percentual de fraude por canal e tipo de credencial (H3). | `fig03_authentication_shield` | Barras agrupadas comparando taxa de fraude em presença física vs. canais digitais, cruzadas com uso de Chip e PIN. |
| **P4** | Disparadas de velocidade e distorções de ticket médio antecipam ataques? | Risco relativo por faixa de `transactions_last_1h` e `amount_vs_avg_ratio` (H4, H5). | `fig04_velocity_amount_anomaly` | Painel duplo lado a lado `1x2` com proporções quadradas idênticas (`set_box_aspect(1)`): (a) Fraude por transações na última hora; (b) Fraude por razão de valor. |
| **P5** | Qual o desempenho preditivo do modelo binário e o ponto de equilíbrio financeiro? | PR-AUC, ROC-AUC e Curva de Custo Financeiro Total ($) (H6). | `fig05_model_performance_cost` | Painel duplo lado a lado `1x2`: (a) Curva Precision-Recall com corte ótimo anotado; (b) Curva de Custo Total ($) vs. Threshold com indicação de $t^*$. |
| **P6** | Quais variáveis mais pesam nas decisões do motor de detecção? | Valores SHAP médios absolutos globais e contribuições locais. | `fig06_shap_explainability` | Gráfico de abelha (*beeswarm*) com as top 15 features por impacto médio absoluto, complementado por waterfall plot para transação de alto risco. |
| **P7** | Qual a capacidade do modelo em tipificar corretamente os 4 tipos de fraude? | Macro F1 e taxa de acerto por modalidade (H7). | `fig07_multiclass_confusion_matrix` | Matriz de confusão normalizada por linha com mapa de calor suave, destacando acertos em diagonais e confusões entre CNP e Synthetic. |
| **P8** | Quão precisa é a estimativa de perda monetária para fraudes confirmadas? | RMSE, MAE e resíduos em instâncias fraudulentas (H8). | `fig08_fraud_amount_residuals` | Gráfico de dispersão entre Perda Real vs. Prevista em escala logarítmica com linha de identidade $y=x$ e histograma marginal de resíduos. |

---

## 6. Requisitos do Sistema

### 6.1 Requisitos Funcionais (RF)
- **RF01 (Ingestão Segura):** Carregar o arquivo `credit_card_fraud_detection_2026.csv` preservando `'N/A'` na coluna `browser` sem conversão espúria para missing value.
- **RF02 (Saneamento Estrutural):** Ordenar estavelmente a base cronologicamente por `transaction_datetime` e descartar as colunas redundantes `transaction_id` e `card_id`.
- **RF03 (Particionamento Temporal):** Dividir a base no regime temporal estrito (Treino até 31/08, Validação até 31/10, Teste até 31/12).
- **RF04 (Modelagem Estágio 1 - Binária):** Treinar e comparar baselines (Regressão Logística ponderada, Random Forest, LightGBM, XGBoost) avaliando PR-AUC e otimizando o limiar financeiro $t^*$.
- **RF05 (Explicabilidade Estágio 1):** Gerar explicações SHAP globais (TreeExplainer) e análises de casos paradigmáticos para auditabilidade regulatória.
- **RF06 (Modelagem Estágio 2 - Multiclasse):** Treinar classificador multiclasse para os 4 subtipos de fraude utilizando balanceamento de classes e otimizando Macro F1.
- **RF07 (Modelagem Estágio 3 - Regressão):** Ajustar regressor condicional sobre fraudes confirmadas para predizer `fraud_amount_lost`, medindo RMSE e MAE.
- **RF08 (Relatório Técnico Quarto):** Compilar a narrativa analítica completa em Quarto (`index.qmd`) com metadados em português (`lang: pt-BR`), `execute: freeze: auto`, tema customizado e publicação automatizada via GitHub Pages.

### 6.2 Requisitos Não-Funcionais (RNF)
- **RNF01 (Reprodutibilidade Estrita):** Fixar semente aleatória (`seed=42`) em todo o ciclo de treino e avaliação.
- **RNF02 (Desempenho de Inferência):** Pipeline de pré-processamento e escoragem com tempo de resposta inferior a 50 ms por transação individual.
- **RNF03 (Isolamento de Alvos):** Garantir via testes automatizados que nenhum modelo de estágio preliminar acesse variáveis de estágios posteriores.
- **RNF04 (Padrão Estético de Figuras):** Exportar todas as figuras estáticas em `dpi=300`, com proporções niveladas, remoção de bordas superiores/direitas (`despine`) e grid sutil (`alpha=0.25`).
- **RNF05 (Manutenibilidade de Código):** Módulos Python em `src/` organizados com tipagem estática, formatação via `ruff` e suíte de testes com `pytest`.

---

## 7. Escopo Negativo (Non-Goals)

Para assegurar foco e entrega rigorosa dentro dos limites desta sprint, estabelecem-se as seguintes exclusões de escopo:
1. **Não construiremos microsserviços ou APIs REST de produção:** O projeto entrega o pipeline analítico completo, código modular versionado, artefatos de modelos treinados e relatório Quarto. Não inclui containers Docker para servir endpoints HTTP em Kubernetes.
2. **Não utilizaremos Deep Learning tabular complexo:** Modelos baseados em gradient boosting de árvores (LightGBM e XGBoost) apresentam eficiência superior em dados tabulares deste perfil, dispensando arquiteturas de redes neurais pesadas.
3. **Não utilizaremos `customer_id` ou `merchant_id` como features categóricas brutas:** O uso direto desses IDs gera memorização espúria e não generaliza para novos clientes/lojistas.
4. **Não utilizaremos validação cruzada k-fold aleatória:** A aleatoriedade viola a causalidade temporal das transações.
5. **Não faremos imputação sintética de nulos na coluna `browser`:** A ausência de navegador em compras presenciais e caixas eletrônicos é uma verdade de negócio e deve ser tratada como categoria nominal própria (`N/A`).
