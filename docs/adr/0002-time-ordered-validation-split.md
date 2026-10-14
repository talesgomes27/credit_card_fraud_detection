# ADR 0002: Estratégia de Partição Temporal Out-of-Time

- **Status:** Aceito
- **Data:** 2026-10-14
- **Contexto:** Transações bancárias possuem forte dependência temporal e dinâmica comportamental continuada. A auditoria estatística do Gate 1 revelou que os 8.000 clientes da base realizam compras distribuídas ao longo de todo o ano de 2026, com uma janela média de atividade de 307 dias. Diversas features de perfil e comportamento (`avg_transaction_amount_30d`, `credit_score`, `account_age_days`, `num_chargebacks_ever`) são estáticas por cliente na base.

## Decisão
Adotar uma partição cronológica estrita *out-of-time* baseada na coluna `transaction_datetime`:
1. **Treino:** Transações entre `2026-01-01` e `2026-08-31` (~66.860 transações, 66,9% da base).
2. **Validação:** Transações entre `2026-09-01` e `2026-10-31` (~16.905 transações, 16,9% da base).
3. **Teste Cego:** Transações entre `2026-11-01` e `2026-12-31` (~16.235 transações, 16,2% da base).

A base de dados será previamente submetida a uma ordenação estável por `transaction_datetime` para corrigir a inversão pontual identificada em 28/02/2026 (linha 16221).

## Alternativas Rejeitadas
- **K-Fold Cross-Validation Aleatório:** Misturaria compras passadas e futuras, vazando o padrão de gastos de um cliente visto no conjunto de teste para o conjunto de treino. Isso inflaria artificialmente as métricas de validação (*lookahead bias*).
- **GroupKFold por `customer_id` Sem Eixo Temporal:** Avalia apenas a generalização para clientes novos e desconhecidos, mas desconsidera a causalidade cronológica de detecção de fraudes em clientes existentes, que constitui o cenário operacional padrão de um banco.

## Consequências
- **Positivas:** Simula a realidade de produção de motores antifraude, onde modelos treinados com o histórico passado operam sobre transações futuras. Garante que os artefatos de pré-processamento (médias, escaladores) sejam calibrados sem vazamento de informação do futuro.
- **Mitigadas:** O conjunto de teste passa a ter tamanho fixo. Para garantir que as métricas no teste cego sejam estatisticamente confiáveis, a partição de 2 meses retém mais de 16.000 transações e ~315 fraudes, volume mais que suficiente para intervalos de confiança estreitos.
