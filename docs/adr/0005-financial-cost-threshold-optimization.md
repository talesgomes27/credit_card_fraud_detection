# ADR 0005: Otimização de Limiar de Decisão por Função de Custo Financeiro Real

- **Status:** Aceito
- **Data:** 2026-10-14
- **Contexto:** Em problemas de classificação desbalanceada com impacto financeiro direto, o limiar padrão de probabilidade de $0,50$ raramente é ideal. Falsos negativos (fraudes não detectadas) custam o montante integral da transação fraudulenta (média de \$136,18, podendo atingir \$4.482,16). Em contrapartida, falsos positivos (transações legítimas bloqueadas) causam custos de atrito e suporte operacional com o cliente (estipulados conservadoramente em \$3,50 por alerta).

## Decisão
Substituir o ponto de corte estático de $0,50$ por uma otimização empírica de limiar $t^* \in (0, 1)$ calibrada no conjunto de **Validação** (Set–Out/2026), minimizando a função de custo financeiro total:

$$C(t) = \sum_{i \in FN(t)} \text{fraud\_amount\_lost}_i + \sum_{i \in FP(t)} C_{\text{atrito}}$$

O ponto ótimo $t^*$ selecionado na Validação será aplicado de forma congelada sobre o conjunto de **Teste Cego** (Nov–Dez/2026) para reportar o valor financeiro poupado em dólares pelo sistema em regime operacional.

## Alternativas Rejeitadas
- **Corte Arbitrário de 0,50:** Ignora a assimetria brutal entre o custo de perder uma fraude de \$500 e o custo de enviar uma mensagem de validação ao cliente de \$3,50.
- **Maximização Pura do F1-Score Convencional:** O F1 trata falsos positivos e falsos negativos com o mesmo peso unitário, sem considerar que uma fraude de \$4.000 é muito mais danosa que uma fraude de \$10.

## Consequências
- **Positivas:** Alinha diretamente o modelo de Machine Learning aos resultados da demonstração de resultados da instituição financeira. Maximiza o ROI da operação de prevenção a fraudes.
- **Mitigadas:** O valor de \$3,50 é uma premissa paramétrica de mercado. O relatório Quarto incluirá uma análise de sensibilidade variando o custo de falso positivo entre \$1,00 e \$10,00 para demonstrar a estabilidade do threshold ótimo.
