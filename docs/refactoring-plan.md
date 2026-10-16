# Plano de Refatoração Cirúrgica em Fatias Verticais (Gate 3)

**Projeto:** Detecção, Tipificação e Mitigação de Fraudes Bancárias (`credit_card_fraud_detection`)  
**Metodologia:** `audit-and-refactoring-pipeline` (Protocolo *Verify-Revert-Stop*)  
**Rede de Segurança:** `tests/characterization/test_pinning_hurdle_pipeline.py` (3 testes / 20 testes totais verdes)  
**Data:** 16 de Outubro de 2026

---

## 1. Princípios de Execução e Guardrails

1. **Protocolo Verify-Revert-Stop:** Cada fatia (*Batch*) será executada atomicamente. Ao final de cada lote, executaremos a suíte completa de 20 testes (`pytest`). Se qualquer teste quebrar de forma não planejada, a alteração será revertida imediatamente antes de qualquer nova tentativa.
2. **Preservação de Integridade Estatística:** Nenhum parâmetro de hiperparametrização ou split temporal do modelo será afrouxado. O conjunto de teste (Nov-Dez/2026) permanece estritamente cego.
3. **Padrão Anti-IA e Estilo Editorial:** A reestruturação de textos em `index.qmd` preservará o tom sênior, técnico e direto, eliminando fórmulas prontas de marketing, falsos contrastes e numeração duplicada de cabeçalhos.

---

## 2. Detalhamento das Fatias Cirúrgicas (*Batches*)

```text
[Rede de Segurança: 20 Testes Verdes]
                  │
                  ▼
Fatia 1: Saneamento Quarto (Navbar, Render Scope & Links) ──→ [Verify: pytest + render]
                  │
                  ▼
Fatia 2: CRISP-ML(Q) em 6 Seções, Baselines & Números     ──→ [Verify: pytest + render]
                  │
                  ▼
Fatia 3: Sintaxe @fig- Nativa Quarto & Silenciamento Warnings─→ [Verify: pytest + ruff]
                  │
                  ▼
Fatia 4: Compilação Estática Final, Auditoria Docs & Handoff ──→ [Zero Regressão / Gate 5]
```

### Fatia 1: Saneamento da Infraestrutura Quarto Website (P1 / AUD-03, AUD-04, AUD-06)
- **Objetivo:** Eliminar a compilação indevida de arquivos de controle (`task.md`), corrigir páginas órfãs na navegação e consertar links externos do GitHub.
- **Arquivos Afetados:** `_quarto.yml`, `docs/03-architecture-and-schema.md`.
- **Ações Técnicas:**
  1. Configurar `project: render:` explícito em `_quarto.yml`, restringindo o build a:
     - `index.qmd`
     - `docs/00_archetype.md`
     - `docs/01-data-discovery-and-contracts.md`
     - `docs/02-PRD-and-specs.md`
     - `docs/03-architecture-and-schema.md`
     - `docs/04-scaffolding-and-conventions.md`
     - `docs/05-implementation-plan.md`
     - `docs/adr/*.md` (ADRs 0001 a 0005)
  2. Reestruturar a `navbar` superior:
     - Menu lateral esquerdo com dropdown de **Arquitetura & Engenharia** (Topologia, Scaffolding e Arquétipo) e dropdown de **Decisões Técnicas (ADRs)** para acesso direto aos 5 registros arquiteturais.
     - Atualizar a `site-url` e o link do ícone GitHub para `https://github.com/talesgomes27/credit_card_fraud_detection`.
  3. Em `docs/03-architecture-and-schema.md`, substituir link local com prefixo `file:///` por referência textual canônica.
  4. Purgar arquivos HTML espúrios de `docs_site/` (`docs_site/task.html`, `docs_site/docs/task.html`).
- **Verificação:** Rodar `quarto render` e verificar ausência de páginas de controle e links quebrados.

---

### Fatia 2: Reestruturação Editorial CRISP-ML(Q) e Tabela de Baselines (P1 / AUD-01, AUD-02)
- **Objetivo:** Organizar o `index.qmd` nas 6 etapas canônicas do padrão corporativo, corrigir a fidelidade numérica de fraudes e introduzir a comparação formal com Baselines.
- **Arquivos Afetados:** `index.qmd`.
- **Ações Técnicas:**
  1. Reorganizar a estrutura de cabeçalhos H2 para as 6 seções canônicas:
     - `## 1. Contexto de Negócio & Hipóteses (Business Understanding / Ask)`
     - `## 2. Auditoria de Dados & Contratos (Data Understanding / Prepare)`
     - `## 3. Análise Exploratória & Engenharia de Atributos (Data Prep & EDA / Process)`
     - `## 4. Modelagem Estatística & Machine Learning (Modeling / Construct)`
     - `## 5. Avaliação, Diagnóstico & Explicabilidade (Evaluation / Share)`
     - `## 6. Recomendações Executivas & Plano de Ação (Deployment / Act)`
  2. Ajustar a Seção 2 com as contagens numéricas reais validadas no Golden Master:
     - Treino: 66.388 linhas, 1.341 fraudes (2,02%).
     - Validação: 16.905 linhas, 362 fraudes (2,14%).
     - Teste Cego: 16.707 linhas, 352 fraudes (2,11%).
     - Total: 2.055 fraudes.
  3. Na Seção 4, introduzir tabela comparativa formal multi-métrica entre os modelos campeões e *Baselines*:
     - **Estágio 1 (Binário):** Baseline Aleatório / Estratificado ($PR-AUC \approx 0,020$; $Custo \approx \$48.000$) vs. LightGBM ($PR-AUC = 0,848$; $Custo = \$13.250$; Economia $> 40\%$).
     - **Estágio 2 (Multiclasse):** Baseline Classe Majoritária ($Macro\text{ }F1 = 0,167$) vs. LightGBM ($Macro\text{ }F1 = 0,477$).
     - **Estágio 3 (Regressão):** Baseline Dummy Mean ($RMSE = \$214,10$; $R^2 = 0,00$) vs. LightGBM ($RMSE = \$63,86$; $R^2 = 0,929$).
- **Verificação:** Conferir fidelidade 1:1 entre o texto, tabelas e código.

---

### Fatia 3: Referências Cruzadas Nativas Quarto e Sanidade de Warnings (P2, P3 / AUD-05, AUD-07, AUD-09)
- **Objetivo:** Adotar a sintaxe de referências nativas do Quarto e silenciar warnings de terceiros nos testes.
- **Arquivos Afetados:** `index.qmd`, `pyproject.toml`.
- **Ações Técnicas:**
  1. No `index.qmd`, substituir os links manuais `[Figura 1](#fig-01)` por `@fig-01` a `@fig-08`, garantindo que os blocos de imagem estejam anotados com `{#fig-01}`, `{#fig-02}`, etc.
  2. No `pyproject.toml`, atualizar `requires-python = ">=3.11"` para refletir o interpretador ativo do ambiente científico.
  3. No `pyproject.toml`, adicionar seção `filterwarnings` para ignorar depreciações conhecidas do `matplotlib` e `shap`, mantendo alertas de código de negócio visíveis.
- **Verificação:** Rodar `pytest` e checar execução 100% limpa sem warnings de fontconfig/pyparsing.

---

### Fatia 4: Verificação de Regressão, Build Final e Handoff (Gate 5)
- **Objetivo:** Garantir zero regressões, recompilar o site estático e preparar o commit de encerramento da auditoria.
- **Ações Técnicas:**
  1. Execução de `pytest` (20 testes verdes).
  2. Execução de `quarto render` gerando `docs_site/` completo e íntegro.
  3. Atualização do `task.md` marcando a conclusão dos Gates 3, 4 e 5.
