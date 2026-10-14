# Pipeline Pré-Código: Detecção de Fraude em Cartão de Crédito

- Arquétipo: A (Data Science, Analytics & ML - Python/Quarto)
- Repositório: `/home/tales/Documentos/github/credit_card_fraud_detection/`
- Fonte de dados: `credit_card_fraud_detection_2026.csv` (100.000 linhas, 43 colunas)

---

## Status dos Gates

- [x] **Gate 1: Contexto Inicial, Descoberta e Contratos de Dados (Ask Preliminar & Prepare)**
  - [x] Inspeção estatística e estrutural do CSV
  - [x] Auditoria do valor 'N/A' em `browser`
  - [x] Verificação empírica dos 6 sinais calibrados de fraude
  - [x] Análise de repetição de clientes e riscos de vazamento temporal
  - [x] Elaboração do contrato de dados em `docs/01-data-discovery-and-contracts.md`
  - [x] Checkpoint humano e decisão de escopo: Arquitetura Hurdle em 3 Estágios + Split Out-of-Time

- [x] **Gate 2: Ideação, Escopo, Hipóteses e Especificação (Ask)**
  - [x] Definição do problema, stakeholders e objetivos operacionais
  - [x] Formulação de hipóteses de negócio e modelagem (H1 a H8)
  - [x] Justificativa matemática das métricas primárias e secundárias por estágio
  - [x] Matriz 1:1 de rastreabilidade (Pergunta Ask ➔ Métrica ➔ Visualização Share)
  - [x] Requisitos funcionais (RF), não-funcionais (RNF) e Non-Goals
  - [x] Elaboração de `docs/02-PRD-and-specs.md`
  - [x] Checkpoint humano e aprovação formal do PRD

- [x] **Gate 3: Arquitetura do Pipeline e Modelagem (Process, Analyze, Share)**
  - [x] Diagrama de fluxo e topologia do pipeline em Mermaid
  - [x] Arquitetura da etapa Process (limpeza, ordenação e engenharia de features)
  - [x] Arquitetura da etapa Analyze (modelagem em 3 estágios, baselines e SHAP)
  - [x] Arquitetura da etapa Share (storyboard visual e regras do Quarto Checklist)
  - [x] Arquitetura da etapa Act (artefatos de modelo, CLI e deploy GitHub Pages)
  - [x] Registros de Decisão Arquitetural (ADRs 0001 a 0005)
  - [x] Elaboração de `docs/03-architecture-and-schema.md`
  - [x] Checkpoint humano e aprovação da arquitetura

- [x] **Gate 4: Scaffolding, Estrutura de Pastas e Convenções**
  - [x] Criação da árvore física de pastas (`src/`, `tests/`, `reports/figures/`, `models/`, `notebooks/`)
  - [x] Configuração do ambiente Python (`pyproject.toml`, `.gitignore`)
  - [x] Configuração do Quarto (`_quarto.yml`, `index.qmd`, tema)
  - [x] Elaboração do guia de convenções e guardrails em `AGENTS.md`
  - [x] Elaboração de `docs/04-scaffolding-and-conventions.md`
  - [x] Checkpoint humano e aprovação do scaffolding

- [x] **Gate 5: Quebra de Tarefas, Estratégia de Testes e Plano de Implementação (Act)**
  - [x] Grafo de dependências e ordem de precedência técnica em Mermaid
  - [x] Estrutura das tarefas atômicas em ciclos TDD (Red-Green-Refactor)
  - [x] Especificação dos testes automatizados e critérios de aceite
  - [x] Comandos de verificação (`pytest`, `ruff`, `quarto render`)
  - [x] Elaboração de `docs/05-implementation-plan.md`
  - [x] Checkpoint humano e autorização final para escrita de código

---

## Ciclo de Implementação (TDD & Execução Atômica)

- [x] **Tarefa 01: Ingestão, Saneamento e Contratos de Dados**
  - [x] Escrever testes em `tests/data/test_loader.py` (Red)
  - [x] Implementar `src/data/loader.py` (Green)
  - [x] Validar conformidade de contratos e commit
- [ ] **Tarefa 02: Particionamento Temporal Out-of-Time**
  - [ ] Escrever testes em `tests/data/test_splitter.py` (Red)
  - [ ] Implementar `src/data/splitter.py` (Green)
  - [ ] Validar ausência de vazamento e commit
- [ ] **Tarefa 03: Pipeline de Engenharia de Features**
  - [ ] Escrever testes em `tests/features/test_pipeline.py` (Red)
  - [ ] Implementar `src/features/pipeline.py` (Green)
  - [ ] Validar fit exclusivo no treino e commit
- [ ] **Tarefa 04: Estágio 1 — Classificador Binário & Curva de Custo**
  - [ ] Escrever testes em `tests/models/test_stage1_binary.py` (Red)
  - [ ] Implementar `src/models/stage1_binary.py` (Green)
  - [ ] Calibrar threshold ótimo $t^*$ e commit
- [ ] **Tarefa 05: Estágio 2 — Classificador Multiclasse de Tipos de Fraude**
  - [ ] Escrever testes em `tests/models/test_stage2_multiclass.py` (Red)
  - [ ] Implementar `src/models/stage2_multiclass.py` (Green)
  - [ ] Validar Macro F1 e commit
- [ ] **Tarefa 06: Estágio 3 — Regressor Condicional de Perda Financeira**
  - [ ] Escrever testes em `tests/models/test_stage3_regressor.py` (Red)
  - [ ] Implementar `src/models/stage3_regressor.py` (Green)
  - [ ] Validar RMSE em fraudes e commit
- [ ] **Tarefa 07: Orquestrador do Hurdle Model (End-to-End)**
  - [ ] Escrever testes em `tests/models/test_predict_pipeline.py` (Red)
  - [ ] Implementar `src/models/predict.py` (Green)
  - [ ] Validar integridade e commit
- [ ] **Tarefa 08: Motor de Explicabilidade SHAP**
  - [ ] Escrever testes em `tests/models/test_explainability.py` (Red)
  - [ ] Implementar `src/models/explainability.py` (Green)
  - [ ] Validar TreeExplainer e commit
- [ ] **Tarefa 09: Storytelling Visual e Geração das 8 Figuras**
  - [ ] Escrever testes em `tests/visualization/test_figures.py` (Red)
  - [ ] Implementar `src/visualization/figures.py` (Green)
  - [ ] Gerar 8 figuras em 300 DPI e commit
- [ ] **Tarefa 10: Compilação do Relatório Quarto e Publicação**
  - [ ] Consolidar métricas e gráficos em `index.qmd`
  - [ ] Renderizar com Quarto 1.9.38 para `docs_site/`
  - [ ] Commit final do relatório e site estático
