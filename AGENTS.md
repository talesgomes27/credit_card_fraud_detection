# Guia Operacional e Guardrails para Agentes de IA (AGENTS.md)

Este documento define os princípios obrigatórios de engenharia, arquitetura e redação para qualquer agente que contribua neste repositório.

---

## 1. Diretrizes de Engenharia e Boas Práticas de Código

1. **Stack Tecnológica:**
   - Python 3.12+ com tipagem estática nos argumentos e retornos de funções.
   - Padrão de estilo e linting: `ruff` (comprimento máximo de linha de 100 caracteres).
   - Suíte de testes automatizados: `pytest`.
   - Visualização e publicação: Quarto (`.qmd`) com metadados em português (`lang: pt-BR`).
2. **Proibição Estrita de Data Leakage:**
   - **Isolamento de Targets:** Nunca inclua `fraud_type` ou `fraud_amount_lost` como variáveis preditivas nos modelos de `is_fraud`.
   - **Isolamento Temporal:** Nunca aplique validação cruzada aleatória (*k-fold* comum). Utilize exclusivamente partição cronológica *out-of-time*.
   - **Fit no Treino:** Ajustadores de escala (*StandardScaler*), codificadores (*TargetEncoder*, *OneHotEncoder*) e imputadores devem executar `.fit()` exclusivamente no conjunto de Treino (`transaction_datetime < '2026-09-01'`). Nos conjuntos de Validação e Teste, aplique apenas `.transform()`.
   - **Descarte de Identificadores:** `transaction_id` e `card_id` devem ser descartados na primeira etapa do pipeline.
3. **Ingestão com Contrato de Dados:**
   - Sempre utilize `keep_default_na=False` ao carregar o CSV de transações para preservar `'N/A'` como categoria explícita válida na coluna `browser`.

---

## 2. Padrões de Storytelling e Visualização (Quarto Quality Checklist)

1. **Prevenção de Numeração Duplicada:**
   - O arquivo `_quarto.yml` está configurado com `number-sections: true`. Portanto, nunca escreva números manualmente nos títulos Markdown (use `## Contexto e Perguntas Norteadoras` em vez de `## 1. Contexto e Perguntas Norteadoras`).
2. **Nivelamento de Subfiguras Lado a Lado:**
   - Ao desenhar painéis compostos `(a)` e `(b)`, gere-os em uma única figura do Matplotlib (`fig, axes = plt.subplots(1, 2, ...)`) aplicando `ax.set_box_aspect(1)` em ambos os eixos.
3. **Estética Gráfica:**
   - Execute sempre `sns.despine()` para eliminar bordas superiores e direitas.
   - Aplique grades sutis com `alpha=0.25` e `linestyle='--'`.
   - Salve todas as figuras em alta resolução a 300 DPI em `reports/figures/` antes de chamar `quarto render`.

---

## 3. Diretrizes de Redação e Autoria (Anti-IA / Humanizer)

1. **Voz Sênior e Pragmática:**
   - Escreva como um engenheiro de machine learning experiente.
   - Evite adjetivação vazia de marketing (*"robusto"*, *"pioneiro"*, *"vital"*, *"orquestração perfeita"*).
2. **Sem Fórmulas Clichês:**
   - Elimine construções do tipo *"não apenas X, mas também Y"*, conclusões em uma linha (*"esse é o verdadeiro ganho"*), travessões duplos (`—`) conectando orações sem critério, ou listas em tríades forçadas.
3. **Variação de Ritmo (Burstiness):**
   - Intercale intencionalmente frases curtas e diretas de 3 a 7 palavras com períodos mais articulados para quebrar a assinatura estatística uniforme típica de LLMs.
4. **Commits Convencionais:**
   - Use mensagens semânticas em minúsculas e imperativas (`feat: ...`, `fix: ...`, `docs: ...`, `refactor: ...`), sem autoelogios promocionais e sem menções a ferramentas de IA.
