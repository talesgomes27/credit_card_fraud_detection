# Detecção, Tipificação e Mitigação de Fraudes Bancárias

Pipeline analítico e preditivo ponta a ponta sobre 100.000 transações de cartão de crédito cobrindo o ano de 2026, estruturado sob a metodologia híbrida **CRISP-ML(Q) + Data Storytelling**.

O sistema emprega uma arquitetura modular em três estágios (*Hurdle Model*):
1. **Estágio 1 (Classificação Binária):** Priorização por PR-AUC com calibragem de limiar financeiro mínimo ($t^* = 0,170$), gerando economia superior a 40% em relação ao corte simétrico de 0,50.
2. **Estágio 2 (Classificação Multiclasse):** Separação de 4 modalidades de ataque (*Card-Not-Present*, *Account-Takeover*, *Synthetic-Identity*, *Card-Present*) com Macro F1 de 0,477.
3. **Estágio 3 (Regressão Condicional):** Estimativa da perda monetária residual não recuperada com RMSE de \$63,86 ($R^2 = 0,929$).

Acesse o relatório executivo completo publicado em:  
👉 **[https://talesgomes27.github.io/credit_card_fraud_detection/](https://talesgomes27.github.io/credit_card_fraud_detection/)**

---

## Estrutura do Repositório

```text
├── credit_card_fraud_detection_2026.csv   # Base bruta (100.000 transações, 43 colunas)
├── index.qmd                              # Relatório executivo Quarto em 6 seções CRISP-ML(Q)
├── _quarto.yml                            # Configuração do Quarto Website e barra de navegação
├── pyproject.toml                         # Especificação PEP 621 e dependências do projeto
├── CHANGELOG.md                           # Histórico de alterações SemVer (Keep a Changelog)
├── docs/                                  # Documentação técnica e governança arquitetural
│   ├── 00_archetype.md                    # Metadados do Arquétipo A (Data Science & Quarto)
│   ├── 01-data-discovery-and-contracts.md # Contratos de dados e perfis estatísticos
│   ├── 02-PRD-and-specs.md                # Especificações de produto e hipóteses de negócio
│   ├── 03-architecture-and-schema.md      # Topologia do pipeline e esquemas
│   ├── 04-scaffolding-and-conventions.md  # Convenções de código e diretórios
│   ├── 05-implementation-plan.md          # Plano de implementação e testes
│   ├── audit-report.md                    # Relatório de auditoria multi-camadas
│   ├── refactoring-plan.md                # Plano de refatoração cirúrgica
│   └── adr/                               # 5 Registros de Decisão Arquitetural (ADRs 0001 a 0005)
├── src/                                   # Código modular de produção
│   ├── data/                              # Ingestão, contratos e split temporal out-of-time
│   ├── features/                          # Engenharia de atributos temporais, espaciais e comportamentais
│   ├── models/                            # Estimadores dos estágios 1, 2 e 3, orquestrador e SHAP
│   └── visualization/                     # Geração das 8 figuras em 300 DPI
├── tests/                                 # Suíte automatizada de testes
│   ├── characterization/                  # Testes Golden Master congelando invariantes
│   ├── data/                              # Testes de ingestão e particionamento
│   ├── features/                          # Testes de transformações de atributos
│   ├── models/                            # Testes de estimadores e pipeline hurdle
│   └── visualization/                     # Testes de integridade visual e DPI
├── reports/figures/                       # 8 figuras canônicas em 300 DPI
└── docs_site/                             # Website estático compilado pelo Quarto
```

---

## Reprodução do Ambiente e Execução

### Pré-requisitos
- Python 3.11+ ou `uv`
- Quarto CLI 1.9+

### Instalação e Testes
```bash
# Instalação das dependências do projeto
pip install .

# Execução da suíte completa de testes (20 testes)
pytest

# Validação de formatação e linting
ruff check src/
```

### Compilação do Relatório Quarto
```bash
# Renderização do site estático em docs_site/
quarto render
```

---

## Licença e Autoria
Desenvolvido por **Tales Gomes**. Disponibilizado sob licença MIT.
