# Relatório de Diagnóstico e Auditoria Técnica (Gate 2)

**Projeto:** Detecção, Tipificação e Mitigação de Fraudes Bancárias (`credit_card_fraud_detection`)  
**Arquétipo:** A — Data Science, Analytics & Quarto  
**Data:** 16 de Outubro de 2026  
**Avaliador:** `audit-and-refactoring-pipeline` (Modo 2 / Voz Sênior Anti-IA)

---

## 1. Sumário Executivo do Diagnóstico

O repositório apresenta maturidade técnica expressiva, contando com 17 testes automatizados verdes, separação modular em `src/`, modelagem em três estágios (*Hurdle Model*) e geração automatizada de 8 figuras de alta resolução (300 DPI). 

Entretanto, a auditoria multi-camadas identificou dívidas técnicas estruturais concentradas em quatro frentes:
1. **Aderência Metodológica ao CRISP-ML(Q):** O relatório principal (`index.qmd`) fundiu as etapas de preparação e processamento em 5 seções em vez das 6 seções canônicas, e não inclui uma tabela multi-métrica comparativa formal contra modelos de *Baseline*.
2. **Integridade de Navegação do Quarto Website:** Documentos centrais de arquitetura (`docs/04-scaffolding-and-conventions.md`, `docs/00_archetype.md` e os 5 Registros de Decisão Arquitetural - ADRs) estão órfãos da barra de navegação, enquanto arquivos de controle interno (`task.md`) foram indevidamente compilados para páginas web públicas.
3. **Sintaxe de Referências Cruzadas:** O `index.qmd` adota links manuais `[Figura 1](#fig-01)` em vez da sintaxe nativa `@fig-01` do Quarto, quebrando a ancoragem automática do motor de renderização.
4. **Governança de Ambiente e Tooling:** Ausência do binário `uv` e de arquivo `uv.lock`, execução sob Python 3.11 quando o manifesto exige `>=3.12`, e falta do plugin `pytest-cov` no ambiente de desenvolvimento ativo.

---

## 2. Matriz de Achados e Triagem de Dívida Técnica

A tabela abaixo classifica todos os achados diagnosticados nas 4 camadas segundo os critérios objetivos de severidade da `severity-matrix-and-triage.md`.

| ID | Linha / Arquivo | Severidade | Diagnóstico Original | Sugestão de Refatoração | Racional Técnico |
| :--- | :--- | :---: | :--- | :--- | :--- |
| `AUD-01` | `index.qmd:42,59` | **Alto (P1)** | Relatório estruturado em 5 seções principais, fundindo *Data Understanding & Contracts (Prepare)* e *Data Prep & EDA (Process)* em uma única seção. | Separar nas 6 seções canônicas do CRISP-ML(Q) + Storytelling: 1. Contexto & Hipóteses; 2. Auditoria & Contratos; 3. EDA & Feature Engineering; 4. Modelagem & Baselines; 5. Avaliação, Diagnóstico & Explicabilidade; 6. Recomendações Executivas & Ação. | Garante conformidade estrita com o padrão corporativo de analytics e paridade metodológica com a `pre-code-pipeline` v2.1. |
| `AUD-02` | `index.qmd:59-84` | **Alto (P1)** | A Seção de Modelagem reporta unicamente os modelos campeões do LightGBM, sem comparação multi-métrica formal contra *Baselines* ingênuos. | Adicionar tabela comparativa formal avaliando Dummy/Baseline Aleatório vs. LightGBM nos 3 estágios (PR-AUC, Macro-F1 e RMSE). | Requisito mandatório de governança científica para justificar a complexidade algorítmica do modelo em produção. |
| `AUD-03` | `_quarto.yml:9-24` | **Alto (P1)** | Navbar incompleta com páginas órfãs (`docs/04-scaffolding-and-conventions.md`, `docs/00_archetype.md` e menu dropdown de ADRs ausentes) e links incorretos de GitHub (`tales` em vez de `talesgomes27`). | Reestruturar a navbar no `_quarto.yml` com menu dropdown para Arquitetura & Decisões (ADRs 0001 a 0005) e corrigir a URL canônica do GitHub para `talesgomes27`. | Elimina páginas mortas e viabiliza a navegação completa da auditoria por revisores externos e liderança técnica. |
| `AUD-04` | `_quarto.yml:1-5` | **Alto (P1)** | Quarto configurado sem escopo de renderização explícito (`project: render:`), causando compilação pública indevida de arquivos de controle (`task.md` e `docs/task.md`). | Declarar explicitamente a lista de arquivos a renderizar em `_quarto.yml` (`index.qmd` e documentos de `docs/`), excluindo arquivos `task.md`. | Evita exposição de artefatos de memória interna e histórico de tarefas na documentação estática de entrega. |
| `AUD-05` | `index.qmd:26-35,92,99...` | **Médio (P2)** | Uso de referências manuais em Markdown `[Figura X](#fig-0X)` em vez de referências cruzadas nativas `@fig-0X`. | Substituir links manuais pela sintaxe nativa `@fig-01` a `@fig-08` do Quarto, ajustando as legendas dos blocos de figura. | Habilita numeração dinâmica do Quarto, navegação por teclado e conformidade com o `quarto-quality-checklist.md`. |
| `AUD-06` | `docs/03-architecture-and-schema.md:27` | **Médio (P2)** | Link absoluto com prefixo `file:///home/tales/.../quarto-quality-checklist.md` inserido no Markdown, quebrando na navegação web estática. | Substituir link absoluto local por citação textual de padrão ou referência canônica interna do projeto. | Evita erro 404 e comportamento inconsistente de navegação no site publicado no GitHub Pages. |
| `AUD-07` | `pyproject.toml:6,21-26` | **Médio (P2)** | Desalinhamento de ambiente: `pyproject.toml` exige Python `>=3.12`, mas runtime ativo é Conda `science` (Python 3.11.16); falta `pytest-cov` instalado no ambiente. | Harmonizar target para `>=3.11` no `pyproject.toml` ou criar ambiente isolado com `uv`, e registrar instruções de medição de cobertura. | Elimina avisos de incompatibilidade de versão e padroniza a reprodutibilidade em CI/CD. |
| `AUD-08` | `tests/` | **Médio (P2)** | Ausência de suíte de testes de caracterização (*Golden Master*) para congelar as predições e métricas do Hurdle Model antes da refatoração. | Criar `tests/characterization/test_golden_master.py` gravando saídas esperadas dos 3 estágios do modelo. | Garante rede de segurança estrita (*Verify-Revert-Stop*) exigida pelo Gate 3 antes de qualquer ajuste de código. |
| `AUD-09` | `tests/data/test_loader.py:..` | **Baixo (P3)** | Warnings de depreciação de bibliotecas externas (`PyparsingDeprecationWarning` do matplotlib e `UserWarning` do LightGBM/SHAP) poluindo o output do pytest. | Adicionar filtro de warnings direcionado no `pyproject.toml` (`[tool.pytest.ini_options] filterwarnings = [...]`). | Mantém o terminal limpo e focado em regressões reais de teste. |

---

## 3. Próximos Passos (Plano de Transição para o Gate 3)

Com a aprovação deste diagnóstico pelo usuário:
1. **No Gate 3:** Construiremos a suíte de caracterização em `tests/characterization/test_golden_master.py` e detalharemos as fatias cirúrgicas de refatoração em `docs/refactoring-plan.md`.
2. **No Gate 4:** Executaremos a refatoração do `index.qmd` nas 6 seções canônicas CRISP-ML(Q), a inserção das tabelas de baseline, a reconfiguração completa da navbar e links do `_quarto.yml`, e a sanitização de warnings.
3. **No Gate 5:** Validaremos a ausência total de regressões via `pytest` e recompilaremos o site Quarto sem links quebrados ou páginas órfãs.
