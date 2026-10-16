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
- [x] **Tarefa 02: Particionamento Temporal Out-of-Time**
  - [x] Escrever testes em `tests/data/test_splitter.py` (Red)
  - [x] Implementar `src/data/splitter.py` (Green)
  - [x] Validar ausência de vazamento e commit
- [x] **Tarefa 03: Pipeline de Engenharia de Features**
  - [x] Escrever testes em `tests/features/test_pipeline.py` (Red)
  - [x] Implementar `src/features/pipeline.py` (Green)
  - [x] Validar fit exclusivo no treino e commit
- [x] **Tarefa 04: Estágio 1 — Classificador Binário & Curva de Custo**
  - [x] Escrever testes em `tests/models/test_stage1_binary.py` (Red)
  - [x] Implementar `src/models/stage1_binary.py` (Green)
  - [x] Calibrar threshold ótimo $t^*$ e commit
- [x] **Tarefa 05: Estágio 2 — Classificador Multiclasse de Tipos de Fraude**
  - [x] Escrever testes em `tests/models/test_stage2_multiclass.py` (Red)
  - [x] Implementar `src/models/stage2_multiclass.py` (Green)
  - [x] Validar Macro F1 e commit
- [x] **Tarefa 06: Estágio 3 — Regressor Condicional de Perda Financeira**
  - [x] Escrever testes em `tests/models/test_stage3_regressor.py` (Red)
  - [x] Implementar `src/models/stage3_regressor.py` (Green)
  - [x] Validar RMSE em fraudes e commit
- [x] **Tarefa 07: Orquestrador do Hurdle Model (End-to-End)**
  - [x] Escrever testes em `tests/models/test_predict_pipeline.py` (Red)
  - [x] Implementar `src/models/predict.py` (Green)
  - [x] Validar integridade e commit
- [x] **Tarefa 08: Motor de Explicabilidade SHAP**
  - [x] Escrever testes em `tests/models/test_explainability.py` (Red)
  - [x] Implementar `src/models/explainability.py` (Green)
  - [x] Validar TreeExplainer e commit
- [x] **Tarefa 09: Storytelling Visual e Geração das 8 Figuras**
  - [x] Escrever testes em `tests/visualization/test_figures.py` (Red)
  - [x] Implementar `src/visualization/figures.py` (Green)
  - [x] Gerar 8 figuras em 300 DPI e commit
- [x] **Tarefa 10: Compilação do Relatório Quarto e Publicação**
  - [x] Consolidar métricas e gráficos em `index.qmd`
  - [x] Renderizar com Quarto 1.9.38 para `docs_site/`
  - [x] Commit final do relatório e site estático

---

## Pipeline de Auditoria e Refatoração (Brownfield / Melhoria Contínua)

- [x] **Gate 1: Reconhecimento, Engenharia Reversa e Enquadramento (Arquétipo A)**
  - [x] Detecção de arquétipo e consolidação de `docs/00_archetype.md` (Arquétipo A)
  - [x] Reconhecimento da topologia existente, árvore de módulos e histórico Git recente
  - [x] Auditoria de governança de dependências e ambiente Python (Conda science / Pytest 9.1 / Ruff 0.4 / Quarto 1.9.38)
  - [x] Verificação do estado inicial de execução da suíte de testes (17/17 passando sem falhas)
  - [x] Checkpoint humano e aprovação do escopo de auditoria
- [x] **Gate 2: Diagnóstico Multi-Camadas Não-Destrutivo (docs/audit-report.md)**
  - [x] Camada de Código & Segurança (dependências, target python, ruff e warnings de bibliotecas)
  - [x] Camada de Testes (17 testes passando, identificada ausência de pytest-cov e caracterização)
  - [x] Camada Analítica, Científica & Visual (aderência a 6 seções CRISP-ML(Q), ausência de baselines comparativos e sintaxe de @fig-)
  - [x] Camada Documental & Editorial (auditoria de navbar, páginas órfãs, ADRs ausentes do menu e exclusão de task.md da renderização pública)
  - [x] Elaboração do relatório formal em `docs/audit-report.md` (9 achados triados: 4 P1, 4 P2, 1 P3)
  - [ ] Checkpoint humano e aprovação de `docs/audit-report.md`
- [x] **Gate 3: Rede de Segurança (Testes de Caracterização) & Plano de Fatias**
  - [x] Construção de testes Golden Master em `tests/characterization/test_pinning_hurdle_pipeline.py` (3 testes / 20 testes totais passando)
  - [x] Congelamento de invariantes de splits, alvos, contagem real de fraudes (2.055) e integridade de figuras
  - [x] Elaboração do plano incremental em `docs/refactoring-plan.md` (4 Fatias: Saneamento Quarto, CRISP-ML(Q), Referências @fig- e Sanidade Final)
  - [ ] Checkpoint humano e autorização para refatoração
- [x] **Gate 4: Execução da Refatoração Cirúrgica Polimórfica (Verify-Revert-Stop)**
  - [x] Fatia 1: Saneamento Quarto (Navbar com dropdowns de ADRs/Scaffolding, escopo de render e correção de links GitHub)
  - [x] Fatia 2: Reestruturação do relatório nas 6 seções canônicas CRISP-ML(Q), tabela comparativa com Baselines e fidelidade numérica
  - [x] Fatia 3: Sintaxe nativa de referências cruzadas `@fig-01` a `@fig-08` e silenciamento de warnings de bibliotecas
  - [x] Fatia 4: Validação integrada contínua e compilação limpa do site Quarto (exit code 0)
- [x] **Gate 5: Verificação de Regressão, Sincronização Documental & Handoff**
  - [x] Execução da suíte completa de 20 testes (100% verdes) e validação de linter `ruff` sem erros
  - [x] Recompilação completa do Quarto (`quarto render`) para `docs_site/` com 12 documentos canônicos
  - [x] Fechamento e parecer de impacto em `docs/audit-report.md` (9/9 dívidas técnicas resolvidas)
  - [x] Handoff pronto para a `release-and-ops-pipeline` (Gate 1)

---

## Pipeline de Release & Ops (Entrega & Publicação)

- [x] **Gate 1: Saneamento de Repositório, Segurança Pré-Push & Lockfiles**
  - [x] Leitura e confirmação de `docs/00_archetype.md` (Arquétipo A)
  - [x] Blindagem do `.gitignore` contra modelos serializados (*.joblib, *.pkl), credenciais (*.env) e builds
  - [x] Varredura de segredos vazados no código e notebooks (0 segredos detectados)
  - [x] Auditoria de arquivos volumosos > 50 MB (nenhum arquivo detectado)
  - [x] Checkpoint humano e aprovação do saneamento pré-push
- [/] **Gate 2: Governança Git, Versionamento Semântico & Changelog**
  - [ ] Determinação de incremento SemVer (`v1.0.0` - Primeiro Release Estável pós-auditoria)
  - [ ] Elaboração de `CHANGELOG.md` seguindo o padrão Keep a Changelog (Modo 1 da writing-pipeline)
  - [ ] Preparação da tag anotada `v1.0.0`
  - [ ] Checkpoint humano e aprovação do changelog e histórico de commits
- [ ] **Gate 3: Automação de CI/CD — GitHub Actions (Arquétipo A: Quarto + GitHub Pages)**
  - [ ] Configuração de `.github/workflows/publish.yml` para publicação contínua no GitHub Pages
  - [ ] Validação do workflow com permissões `pages: write` e `id-token: write`
  - [ ] Checkpoint humano e aprovação do workflow de CI/CD
- [ ] **Gate 4: Deploy, Publicação e Empacotamento (Polimórfico - GitHub Pages)**
  - [ ] Validação de `.nojekyll` em `docs_site/` para compatibilidade com GitHub Pages
  - [ ] Execução de git push autorizado para o repositório remoto
  - [ ] Ativação da publicação no GitHub Pages
- [ ] **Gate 5: Smoke Tests Pós-Deploy & Homologação Final**
  - [ ] Verificação da URL pública do site (`https://talesgomes27.github.io/credit_card_fraud_detection`)
  - [ ] Auditoria de HTTP status 200, links internos da navbar e integridade de carregamento das figuras
  - [ ] Emissão do parecer final de entrega homologada
