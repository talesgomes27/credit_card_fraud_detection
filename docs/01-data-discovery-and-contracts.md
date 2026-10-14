# Descoberta de Dados, Auditoria e Contratos de Dados (Gate 1)

Documento técnico de auditoria preliminar (*Ask* preliminar & *Prepare*) para o projeto de detecção de fraudes em transações de cartão de crédito.
Base de dados auditada: `credit_card_fraud_detection_2026.csv`.

---

## 1. Contexto de Negócio e Síntese Inicial (Ask Preliminar)

Sistemas legados de detecção de fraude frequentemente enfrentam dois gargalos: datasets públicos excessivamente anonimizados (como o clássico Kaggle 2013 com variáveis PCA `V1` a `V28`) e a falta de interpretabilidade para regras de negócio operacionais.

O dataset inspecionado resolve essa limitação. Ele disponibiliza 43 variáveis com nomes funcionais diretos e distribuições calibradas contra relatórios de referência da indústria:
- Federal Reserve Payments Study (2023)
- Nilson Report (2023)
- FICO Falcon Fraud Manager Benchmarks
- European Central Bank Card Fraud Report (2022)
- TransUnion Global Fraud and Identity Report (2023)

O problema analítico abrange três camadas com níveis distintos de complexidade:
1. **Detecção Binária (`is_fraud`):** Decisão transacional em tempo real (fraude vs. legítima), com classe minoritária severamente desbalanceada (~2,06%).
2. **Tipificação de Fraude (`fraud_type`):** Classificação multiclasse em 5 categorias (Legítima, Card-Not-Present, Account-Takeover, Synthetic-Identity, Card-Present).
3. **Quantificação de Perda Financeira (`fraud_amount_lost`):** Estimação de valor residual não recuperado após estorno parcial (*zero-inflated continuous*).

---

## 2. Diagnóstico da Fonte e Profiling Estatístico (Fase Prepare)

### 2.1 Metadados e Volumetria Geral

A inspeção direta do arquivo `credit_card_fraud_detection_2026.csv` confirmou:
- **Linhas:** 100.000 registros.
- **Colunas:** 43 atributos (40 explicativos e 3 variáveis target).
- **Duplicatas exatas:** 0 linhas.
- **Duplicatas de `transaction_id`:** 0 linhas.
- **Tamanho do arquivo em disco:** 25,7 MB.
- **Clientes únicos (`customer_id`):** 8.000 clientes.
- **Cartões únicos (`card_id`):** 8.000 cartões (relação exata de 1 cartão por cliente).
- **Lojistas únicos (`merchant_id`):** 3.000 estabelecimentos.
- **Janela temporal real (`transaction_datetime`):** `2026-01-01 02:28:51` até `2026-12-31 23:45:47` (ano fiscal 2026 completo, apesar da menção histórica a 2024 no texto descritivo do `about.txt`).

### 2.2 Auditoria Crítica: Tratamento de `'N/A'` na Coluna `browser`

O parser padrão de bibliotecas como pandas (`pd.read_csv`) inclui a string `'N/A'` na lista default de valores nulos (`_NA_VALUES`). Quando o CSV é carregado sem parâmetros adicionais, pandas marca 41.709 registros como `NaN` na coluna `browser`.

Essa conversão é um falso positivo de ausência de dados.
A análise cruzada entre canais de transação e navegadores demonstra que o valor `'N/A'` é uma categoria válida do domínio:

| Canal (`transaction_channel`) | `App` | `Chrome` | `Edge` | `Firefox` | `N/A` | `Other` | `Safari` | Total |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **ATM** | 0 | 0 | 0 | 0 | **5.932** | 0 | 0 | 5.932 |
| **In-store** | 0 | 0 | 0 | 0 | **35.777** | 0 | 0 | 35.777 |
| **Mobile App** | **12.759** | 0 | 0 | 0 | 0 | 0 | 0 | 12.759 |
| **Online** | 0 | 20.359 | 4.544 | 5.448 | 0 | 2.340 | 12.841 | 45.532 |
| **Total Geral** | 12.759 | 20.359 | 4.544 | 5.448 | 41.709 | 2.340 | 12.841 | 100.000 |

Transações físicas presenciais (*In-store*) e operações em terminais de autoatendimento (*ATM*) não utilizam navegador web. Por definição, não possuem browser. Tratar esses registros como missing desnecessário gera erros em imputações automáticas.
**Regra obrigatória de leitura no pipeline:** carregar o CSV com `keep_default_na=False` ou mapear explicitamente `na_values=['']` para manter `'N/A'` como categoria literal estruturada (`'Not Applicable'`).

### 2.3 Auditoria Temporal e Inversão em 28/02/2026

O dataset está quase totalmente ordenado cronologicamente, com uma única exceção identificada entre as linhas 16220 e 16221:
- Linha 16220: `TXN-00016221` em `2026-02-28 23:56:58`
- Linha 16221: `TXN-00016222` em `2026-02-28 00:21:25`

O dia 28 de fevereiro de 2026 contém 579 transações particionadas em dois blocos sequenciais concatenados.
Para garantir o particionamento temporal rigoroso no Gate 3 e Gate 5, o carregamento inicial deve executar ordenação estável por `transaction_datetime`.

---

## 3. Catálogo Detalhado das 43 Variáveis

As 43 colunas foram validadas e organizadas em seus 7 grupos funcionais de entrada mais o grupo de variáveis target.

### 3.1 Identificadores de Transação (5 variáveis)

| Coluna | Tipo Físico | Cardinalidade | Domínio / Padrão | Descrição e Papel Analítico |
| :--- | :--- | :--- | :--- | :--- |
| `transaction_id` | String | 100.000 | `TXN-00000001` a `TXN-00100000` | Identificador único de transação. Excluir das features de modelagem. |
| `customer_id` | String | 8.000 | `CUS-00000001` a `CUS-00008000` | Identificador do cliente. Média de 12,5 transações/cliente. Excluir como feature direta (evitar overfitting). |
| `card_id` | String | 8.000 | `CRD-00000001` a `CRD-00008000` | Identificador do cartão. 100% redundante com `customer_id`. Excluir. |
| `merchant_id` | String | 3.000 | `MER-00000001` a `MER-00003000` | Identificador do lojista. 15 categorias em 20 países. Avaliar target encoding ou agregação. |
| `transaction_datetime` | String (ISO) | 99.762 | `YYYY-MM-DD HH:MM:SS` | Timestamp da transação. Base para split temporal out-of-time. |

### 3.2 Detalhes da Transação (6 variáveis)

| Coluna | Tipo Físico | Mín | Média / Pct | Máx | Descrição |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `transaction_amount` | Float64 | \$0,50 | \$126,87 | \$15.000,00 | Valor monetário em USD. Distribuição assimétrica à direita (mediana \$52,26). |
| `merchant_category` | String | 15 cats | Moda, Varejo, etc. | 15 cats | Setor comercial do lojista. Maior volume: *Online Retail* (13,4%) e *Restaurant* (12,7%). |
| `merchant_country` | String | 20 países | USA (44,5%), UK (8,3%) | 20 países | País do estabelecimento comercial. |
| `transaction_hour` | Int64 | 0 | 14,39 | 23 | Hora do dia (0 a 23). Madrugada (01h–04h) concentra taxa de risco anômala. |
| `transaction_day_of_week` | String | 7 dias | Segunda a Domingo | 7 dias | Dia da semana. Distribuição balanceada (~14,2% por dia). |
| `card_present` | Int64 | 0 | 35,78% | 1 | Indicador binário de presença física do cartão no PDV. 1 = *In-store*. |

### 3.3 Perfil do Cliente (7 variáveis - atributos estáticos por cliente)

| Coluna | Tipo Físico | Mín | Média | Máx | Descrição |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `customer_age` | Int64 | 18 | 42,2 | 85 | Idade do titular. Constante por `customer_id`. |
| `customer_gender` | String | 3 cats | Female (50,6%), Male (47,6%), Other (1,8%) | Gênero declarado no cadastro do cliente. |
| `customer_state` | String | 40 estados | CT, NE, AL, SC, VA, etc. | Estado de residência do cliente nos EUA. |
| `account_age_days` | Int64 | 1 | 785,5 | 3.650 | Dias desde abertura da conta. Mediana 554 dias. |
| `credit_limit` | Float64 | \$500,00 | \$10.083,18 | \$36.670,00 | Limite total do cartão de crédito em USD. |
| `credit_score` | Int64 | 328 | 682,3 | 850 | Escore de crédito estilo FICO. Mediana 683. |
| `num_cards_held` | Int64 | 1 | 2,2 | 5 | Quantidade de cartões ativos em nome do titular. |

### 3.4 Features de Velocidade (6 variáveis - dinâmicas por transação)

| Coluna | Tipo Físico | Mín | Mediana | Máx | Descrição |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `transactions_last_1h` | Int64 | 0 | 0 | 8 | Contagem de transações do cliente na última hora. |
| `transactions_last_24h` | Int64 | 0 | 2 | 18 | Contagem de transações nas últimas 24 horas. |
| `transactions_last_7d` | Int64 | 0 | 7 | 42 | Contagem de transações nos últimos 7 dias. |
| `amount_last_1h` | Float64 | \$0,00 | \$0,00 | \$11.421,39 | Volume financeiro acumulado na última hora em USD. |
| `amount_last_24h` | Float64 | \$0,00 | \$197,70 | \$15.161,52 | Volume financeiro acumulado nas últimas 24 horas em USD. |
| `days_since_last_transaction` | Float64 | 0,0 | 2,4 | 47,3 | Intervalo em dias desde a transação imediatamente anterior. |

### 3.5 Sinais de Risco (7 variáveis)

| Coluna | Tipo Físico | Mín | Média | Máx | Descrição |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `distance_from_home_km` | Float64 | 0,0 | 4.316,8 km | 14.999,8 km | Distância entre endereço cadastral e lojista. |
| `distance_from_last_tx_km` | Float64 | 0,0 | 939,4 km | 11.999,6 km | Salto espacial desde a transação anterior. Mediana 13,3 km. |
| `international_transaction` | Int64 | 0 | 55,48% | 1 | 1 se lojista fora dos EUA (`merchant_country != 'USA'`). |
| `high_risk_country` | Int64 | 0 | 4,93% | 1 | 1 se país pertence ao quarteto de risco (China, Nigéria, Romênia, Rússia). |
| `amount_vs_avg_ratio` | Float64 | 0,001 | 1,96 | 686,32 | Razão entre valor atual e média de 30 dias do cliente. |
| `used_chip` | Int64 | 0 | 31,48% | 1 | 1 se transação utilizou chip EMV físico. |
| `used_pin` | Int64 | 0 | 18,39% | 1 | 1 se autenticação por senha numérica (PIN) foi fornecida. |

### 3.6 Dispositivo e Canal (4 variáveis)

| Coluna | Tipo Físico | Cardinalidade | Domínio / Categorias | Descrição |
| :--- | :--- | :--- | :--- | :--- |
| `transaction_channel` | String | 4 | `Online`, `In-store`, `Mobile App`, `ATM` | Meio de realização da compra. |
| `device_type` | String | 4 | `Mobile`, `POS Terminal`, `Desktop`, `ATM` | Equipamento físico/lógico utilizado na operação. |
| `browser` | String | 7 | `N/A`, `Chrome`, `Safari`, `App`, `Firefox`, `Edge`, `Other` | Navegador web ou aplicativo de origem. |
| `is_weekend` | Int64 | 2 | 0 (dia útil: 71,6%), 1 (fim de semana: 28,4%) | Flag binária de sábado ou domingo. |

### 3.7 Comportamento de Conta (5 variáveis)

| Coluna | Tipo Físico | Mín | Média | Máx | Descrição |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `avg_transaction_amount_30d` | Float64 | \$5,38 | \$124,40 | \$2.000,00 | Ticket médio do cliente no histórico recente (estático por cliente). |
| `std_transaction_amount_30d` | Float64 | \$2,46 | \$68,46 | \$1.334,92 | Desvio padrão histórico dos gastos (estático por cliente). |
| `num_declined_last_30d` | Int64 | 0 | 0,51 | 5 | Transações negadas nos últimos 30 dias (dinâmica por transação). |
| `num_chargebacks_ever` | Int64 | 0 | 0,37 | 4 | Total acumulado de contestações históricas (estático por cliente). |
| `email_domain` | String | 5 | `gmail`, `corporate`, `yahoo`, `hotmail`, `other` | Domínio de e-mail registrado na instituição emissora. |

### 3.8 Variáveis Target Coexistentes (3 variáveis)

| Coluna | Tipo Físico | Classes / Domínio | Distribuição Global | Comportamento Condicional |
| :--- | :--- | :--- | :--- | :--- |
| `is_fraud` | Int64 | `0` (legítima), `1` (fraude) | 0: 97.945 (97,95%)<br>1: 2.055 (2,055%) | Alvo primário para classificação binária supervisionada. |
| `fraud_type` | String | 5 categorias nominais | Legitimate: 97.945 (97,95%)<br>Card-Not-Present: 1.054 (1,05%)<br>Account-Takeover: 447 (0,45%)<br>Synthetic-Identity: 309 (0,31%)<br>Card-Present: 245 (0,24%) | Entre fraudes (`is_fraud == 1`):<br>- Card-Not-Present: 51,29%<br>- Account-Takeover: 21,75%<br>- Synthetic-Identity: 15,04%<br>- Card-Present: 11,92% |
| `fraud_amount_lost` | Float64 | Contínua não-negativa | Média total: \$2,80<br>Desvio total: \$43,67 | Em legítimas (`is_fraud == 0`): \$0,00 rigoroso.<br>Em fraudes (`is_fraud == 1`):<br>- Mín: \$0,58<br>- Mediana: \$50,09<br>- Média: \$136,18<br>- Máx: \$4.482,16 |

---

## 4. Confirmação Empírica dos 6 Sinais Calibrados de Fraude

Os 6 sinais analíticos citados na documentação foram testados contra a base bruta. Os multiplicadores observados comprovam o alinhamento estatístico:

| Sinal Calibrado | Segmento Auditado | Taxa de Fraude | Baseline Geral | Multiplicador Observado |
| :--- | :--- | :--- | :--- | :--- |
| **1. Horário Noturno** | `transaction_hour` em `[1, 2, 3, 4]` | **8,52%** | 2,05% | **4,15x** |
| **2. País de Alto Risco** | `merchant_country` em `['China', 'Nigeria', 'Romania', 'Russia']` | **11,05%** | 2,05% | **5,38x** |
| **3. Conta Recente** | `account_age_days < 90` | **4,87%** | 2,05% | **2,37x** |
| **4. Rajada de Velocidade** | `transactions_last_1h >= 3` | **5,57%** | 2,05% | **2,71x** |
| **5. Autenticação EMV (Chip/PIN)** | `used_chip == 1` e `used_pin == 1` | **0,49%** (0,75% chip / 0,77% pin) | 2,05% | **0,24x** (redução protetiva) |
| **6. Categoria Online Retail** | `merchant_category == 'Online Retail'` | **3,32%** (seguida de Travel com 2,92%) | 2,05% | **1,62x** |

---

## 5. Análise de Risco de Data Leakage e Estratégia de Partição

### 5.1 Isolamento Rigoroso entre Targets
As três variáveis target possuem dependência determinística:
- `fraud_amount_lost` é maior que zero se, e somente se, `is_fraud == 1`.
- `fraud_type` é diferente de `'Legitimate'` se, e somente se, `is_fraud == 1`.
- Não existem contradições lógicas na base: 0 inconsistências detectadas em 100.000 linhas.

**Contrato de isolamento de features:**
1. Nenhum modelo treinado para estimar `is_fraud` pode receber `fraud_type` ou `fraud_amount_lost`.
2. Modelos preditivos de `fraud_type` ou de perda financeira devem operar sob arquitetura desacoplada (classificação direta com exclusão mútua das outras targets ou arquitetura em dois estágios / modelo *hurdle*).

### 5.2 Exclusão de Identificadores Puros e Redundâncias
- `transaction_id`: chave sequencial unívoca sem poder generalizável. Exclusão mandatória.
- `card_id`: possui mapeamento bijetor com `customer_id` (`CRD-XXXX` coincide com `CUS-XXXX`). Não agrega valor analítico. Exclusão mandatória.
- `customer_id`: alta cardinalidade (8.000 IDs). Se mantido cru, gera memorização indevida de histórico por cliente em vez de padrões generalizáveis de fraude.

### 5.3 Vazamento Temporal e Estrutura de Clientes
A análise de dispersão temporal revelou que os clientes realizam transações distribuídas ao longo de todo o ano. A janela média de atividade por cliente é de 307 dias.
Além disso, certas features de perfil e comportamento (`avg_transaction_amount_30d`, `std_transaction_amount_30d`, `credit_score`, `account_age_days`, `num_chargebacks_ever`) são estáticas por cliente na base.

**Impacto arquitetural:**
- Uma validação aleatória cruzada (*K-Fold* clássico) misturaria transações do mesmo cliente no treino e teste. O modelo aprenderia a memorizar perfis de clientes específicos em vez de detectar anomalias transacionais.
- Para simular a produção de um motor antifraude bancário, a validação deve adotar **Time-Ordered Split** (divisão temporal out-of-time):
  - **Treino:** Transações cronológicas de Janeiro a Agosto de 2026 (~66.800 transações).
  - **Validação:** Transações de Setembro a Outubro de 2026 (~16.900 transações).
  - **Teste Cego:** Transações de Novembro a Dezembro de 2026 (~16.300 transações).

---

## 6. Contratos Formais de Qualidade de Dados (Data Contracts)

Para impedir a propagação de dados corrompidos ou mal formatados para as fases de processamento e modelagem, definem-se os seguintes contratos formais.

### Contrato 1: Integridade Estrutural e Schema
```yaml
schema_contract:
  expected_rows: 100000
  expected_columns: 43
  allow_nulls: false # Zero nulos permitidos após leitura com keep_default_na=False
  id_uniqueness:
    transaction_id: unique
    card_id_count: 8000
    customer_id_count: 8000
    merchant_id_count: 3000
```

### Contrato 2: Domínios Categóricos Válidos
```yaml
domain_contract:
  merchant_category:
    allowed:
      - "Auto & Transportation"
      - "Clothing & Apparel"
      - "Education"
      - "Electronics & Technology"
      - "Entertainment & Streaming"
      - "Financial Services"
      - "Gas Station"
      - "Grocery & Supermarket"
      - "Healthcare & Pharmacy"
      - "Home Improvement"
      - "Hotel & Lodging"
      - "Online Retail"
      - "Restaurant & Dining"
      - "Travel & Airlines"
      - "Utilities & Telecom"
  transaction_channel:
    allowed: ["ATM", "In-store", "Mobile App", "Online"]
  device_type:
    allowed: ["ATM", "Desktop", "Mobile", "POS Terminal"]
  browser:
    allowed: ["App", "Chrome", "Edge", "Firefox", "N/A", "Other", "Safari"]
  customer_gender:
    allowed: ["Female", "Male", "Other"]
  email_domain:
    allowed: ["corporate", "gmail", "hotmail", "other", "yahoo"]
  fraud_type:
    allowed:
      - "Legitimate"
      - "Card-Not-Present"
      - "Account-Takeover"
      - "Synthetic-Identity"
      - "Card-Present"
```

### Contrato 3: Limites Numéricos e Invariantes Físicas
```yaml
range_and_invariance_contract:
  ranges:
    transaction_amount: { min: 0.50, max: 20000.00 }
    transaction_hour: { min: 0, max: 23 }
    customer_age: { min: 18, max: 100 }
    credit_score: { min: 300, max: 850 }
    credit_limit: { min: 500.00, max: 50000.00 }
    amount_vs_avg_ratio: { min: 0.0, max: 1000.0 }
    transactions_last_1h: { min: 0, max: 20 }
    fraud_amount_lost: { min: 0.0, max: 15000.00 }
  invariants:
    # 1. Regra de presença física do cartão
    - condition: "card_present == 1"
      must_match: "transaction_channel == 'In-store' and device_type == 'POS Terminal' and browser == 'N/A'"
    # 2. Regra de ATM
    - condition: "transaction_channel == 'ATM'"
      must_match: "device_type == 'ATM' and browser == 'N/A' and card_present == 0"
    # 3. Consistência entre canais móveis e browser
    - condition: "transaction_channel == 'Mobile App'"
      must_match: "browser == 'App' and device_type == 'Mobile'"
    # 4. Consistência de alvos de fraude
    - condition: "is_fraud == 0"
      must_match: "fraud_type == 'Legitimate' and fraud_amount_lost == 0.0"
    - condition: "is_fraud == 1"
      must_match: "fraud_type != 'Legitimate' and fraud_amount_lost > 0.0"
    # 5. Regra de país de alto risco
    - condition: "high_risk_country == 1"
      must_match: "merchant_country in ['China', 'Nigeria', 'Romania', 'Russia']"
    # 6. Regra de transação internacional
    - condition: "international_transaction == 1"
      must_match: "merchant_country != 'USA'"
```

---

## 7. Conclusão do Gate 1 e Próximos Passos

A base `credit_card_fraud_detection_2026.csv` possui consistência estrutural impecável, livre de ruídos acidentais e aderente aos benchmarks econômicos da literatura.
O principal ponto de atenção técnica na ingestão é a preservação de `'N/A'` como string válida na coluna `browser`.

Com este diagnóstico concluído e contratualizado, o projeto cumpre os requisitos de entrada para a transição ao **Gate 2 (Ideação, Escopo, Hipóteses e Especificação)**.
