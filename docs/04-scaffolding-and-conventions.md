---
title: "Scaffolding e Convenções (Gate 4)"
subtitle: "Estrutura física de pastas, convenções e guardrails técnicos"
author: "Tales"
date: "2026-10-14"
---

- **Projeto:** Detecção, Tipificação e Quantificação de Fraudes em Cartão de Crédito
- **Arquétipo:** A — Data Science, Analytics & Machine Learning
- **Artefato:** `docs/04-scaffolding-and-conventions.md`

---

## 1. Árvore Física do Repositório

A estrutura de diretórios foi instanciada para isolar responsabilidades e garantir reproducibilidade analítica:

```text
credit_card_fraud_detection/
├── .gitignore                                   # Filtros de arquivos ignorados (cache, venv, quarto)
├── _quarto.yml                                  # Configuração do site e do relatório Quarto
├── index.qmd                                    # Documento principal do relatório executivo
├── pyproject.toml                               # Dependências, metadados de build, ruff e pytest
├── AGENTS.md                                    # Guardrails e regras operacionais para agentes de IA
├── task.md                                      # Rastreador de progresso dos 5 Gates
├── about.txt                                    # Documentação descritiva da base bruta
├── credit-card-fraud-detection-dataset-2026-metadata.json # Metadados Croissant Kaggle
├── credit_card_fraud_detection_2026.csv         # Fonte bruta (100k linhas x 43 colunas)
├── docs/                                        # Artefatos formais do pipeline pré-código
│   ├── 01-data-discovery-and-contracts.md       # Gate 1: Auditoria e contratos de dados
│   ├── 02-PRD-and-specs.md                      # Gate 2: PRD, hipóteses e matriz Ask-Share
│   ├── 03-architecture-and-schema.md            # Gate 3: Topologia técnica e storyboard visual
│   ├── 04-scaffolding-and-conventions.md        # Gate 4: Árvore física e convenções (este doc)
│   ├── task.md                                  # Cópia persistente do status dos Gates
│   └── adr/                                     # Registros de Decisão Arquitetural
│       ├── 0001-hurdle-three-stage-architecture.md
│       ├── 0002-time-ordered-validation-split.md
│       ├── 0003-tabular-gradient-boosting-vs-deep-learning.md
│       ├── 0004-browser-na-preservation-and-data-contract.md
│       └── 0005-financial-cost-threshold-optimization.md
├── src/                                         # Código-fonte modular da aplicação
│   ├── __init__.py
│   ├── data/                                    # Ingestão, particionamento e contratos
│   │   └── __init__.py
│   ├── features/                                # Pipeline de transformação e feature engineering
│   │   └── __init__.py
│   ├── models/                                  # Treinamento, calibração e inferência Hurdle
│   │   └── __init__.py
│   └── visualization/                           # Geração das 8 figuras e storytelling
│       └── __init__.py
├── tests/                                       # Suíte de testes automatizados (pytest)
│   └── __init__.py
├── reports/                                     # Artefatos visuais gerados
│   └── figures/                                 # Figuras em alta resolução (300 DPI SVG/PNG)
├── models/                                      # Modelos e pipelines serializados (.joblib)
└── notebooks/                                   # Análises exploratórias pontuais
```

---

## 2. Convenções e Padrões Técnicos Adotados

### 2.1 Padrões de Código Python
- **Versão Base:** Python 3.12+ com anotações de tipo explícitas (`from typing import ...`).
- **Formatação e Linter:** `ruff` com limite estrito de 100 caracteres por linha.
- **Imports:** Ordenação padronizada automática via regras `I` (isort) do ruff.
- **Tratamento de Exceções:** Validação de pré-condições via asserções e erros específicos (`ValueError`, `FileNotFoundError`), sem blocos `except Exception: pass` silenciosos.

### 2.2 Estrutura Modular em `src/`
- **`src/data/`:**
  - `loader.py`: Função `load_clean_data()` com `keep_default_na=False` e ordenação estável.
  - `splitter.py`: Função `time_ordered_split()` que divide em Treino, Validação e Teste Cego.
  - `contracts.py`: Validador automatizado das regras definidas no Gate 1.
- **`src/features/`:**
  - `pipeline.py`: Pipeline customizado do Scikit-Learn que executa transformações cíclicas, espaciais, de velocidade e de ticket médio, com `fit` exclusivo no treino.
- **`src/models/`:**
  - `stage1_binary.py`: Treinamento do LightGBM/XGBoost para `is_fraud`, varredura de threshold financeiro e cálculo de PR-AUC.
  - `stage2_multiclass.py`: Classificador para `fraud_type` com ponderação de classes.
  - `stage3_regressor.py`: Regressor condicional com Huber Loss para `fraud_amount_lost`.
  - `predict.py`: Classe `FraudDetectionHurdlePipeline` que consolida os três estágios em uma única chamada `predict_transactions()`.
  - `explainability.py`: Wrapper do `shap.TreeExplainer` para extração de importâncias globais e waterfall plots locais.
- **`src/visualization/`:**
  - `figures.py`: Funções geradoras das figuras `fig01` a `fig08`, aplicando `sns.despine()`, `set_box_aspect(1)` e exportação para `reports/figures/`.

---

## 3. Padrões de Storytelling Quarto e Publicação

- **Metadados:** `lang: pt-BR` no arquivo `_quarto.yml`. Rótulos de navegação e legendas em português nativo.
- **Cabeçalhos:** O `_quarto.yml` ativa `number-sections: true`. Títulos em `index.qmd` utilizam nomes sem numeração manual para evitar repetição (`1. 1. Título`).
- **Cache de Execução:** `execute: freeze: auto` ativado para isolar compilações estéticas do tempo de treinamento de modelos.
- **Deploy:** A pasta de saída configurada é `docs_site`, mantendo a pasta `docs/` intacta para as especificações de engenharia do repositório.

---

## 4. Conclusão do Gate 4 e Próximos Passos

O scaffolding do projeto está instanciado, configurado e documentado.
Com a aprovação deste Gate, passamos ao **Gate 5: Quebra de Tarefas, Estratégia de Testes e Plano de Implementação**, no qual desenharemos o grafo atômico de tarefas ordenadas por dependência com comandos exatos de verificação automatizada (*TDD*).
