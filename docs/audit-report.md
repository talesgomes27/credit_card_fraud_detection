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

## 3. Resolução Cirúrgica e Validação de Dívidas Técnicas (Gates 3 a 5)

Todas as 9 dívidas técnicas triadas no Gate 2 foram resolvidas atomicamente sob o protocolo *Verify-Revert-Stop*:

| ID | Severidade | Ação Executada | Status Pós-Refatoração |
| :--- | :---: | :--- | :---: |
| `AUD-01` | **Alto (P1)** | Reorganização do `index.qmd` nas 6 etapas canônicas do CRISP-ML(Q) + Storytelling. | **Resolvido** |
| `AUD-02` | **Alto (P1)** | Inserção de tabela comparativa multi-métrica contra *Baselines* ingênuos na Seção 4. | **Resolvido** |
| `AUD-03` | **Alto (P1)** | Reestruturação completa da navbar no `_quarto.yml` com menus de Arquitetura e ADRs (0001 a 0005) e correção da URL do GitHub (`talesgomes27`). | **Resolvido** |
| `AUD-04` | **Alto (P1)** | Configuração de `project: render:` explícito no `_quarto.yml`, expurgando arquivos de memória interna (`task.md`) do build estático. | **Resolvido** |
| `AUD-05` | **Médio (P2)** | Migração integral para referências cruzadas nativas do Quarto (`@fig-01` a `@fig-08`). | **Resolvido** |
| `AUD-06` | **Médio (P2)** | Saneamento de link absoluto local `file:///` em `docs/03-architecture-and-schema.md`. | **Resolvido** |
| `AUD-07` | **Médio (P2)** | Harmonização de `requires-python = ">=3.11"` e `target-version = "py311"` no `pyproject.toml`. | **Resolvido** |
| `AUD-08` | **Médio (P2)** | Implementação da suíte de caracterização em `tests/characterization/test_pinning_hurdle_pipeline.py`. | **Resolvido** |
| `AUD-09` | **Baixo (P3)** | Silenciamento seletivo de warnings de bibliotecas terceiras no `pytest.ini_options`. | **Resolvido** |

---

## 4. Métricas de Impacto e Parecer Técnico de Conclusão

- **Dívidas Técnicas Eliminadas:** 9/9 (100% de cobertura dos achados triados).
- **Métricas de Teste:** 20 testes verdes (3 testes de caracterização + 17 testes de unidade/integração).
- **Integridade Quarto Website:** 12 documentos compilados com exit code `0`, sem links quebrados, páginas órfãs ou arquivos de controle expostos.
- **Handoff Recomendado:** O repositório encontra-se saneado, documentado e testado, pronto para ingestão pela **`release-and-ops-pipeline`** para publicação no GitHub Pages.
