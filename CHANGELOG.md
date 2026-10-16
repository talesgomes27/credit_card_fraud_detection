# Changelog

Todas as alterações notáveis neste projeto serão documentadas neste arquivo.

O formato é baseado em [Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/)
e este projeto adere ao [Versionamento Semântico](https://semver.org/lang/pt-BR/).

---

## [1.0.0] - 2026-10-16

### Adicionado
- **Pipeline Hurdle em Três Estágios:**
  - Estágio 1: Classificador binário LightGBM otimizado por custo financeiro ($t^* = 0,170$), gerando economia superior a 40% em relação ao ponto de corte tradicional de 0,50.
  - Estágio 2: Classificador multiclasse discriminando 4 subtipos de fraude (*Card-Not-Present*, *Account-Takeover*, *Synthetic-Identity*, *Card-Present*) com Macro F1-Score de 0,477.
  - Estágio 3: Regressor condicional com LightGBM estimando o prejuízo financeiro residual não recuperado com RMSE de \$63,86 ($R^2 = 0,929$).
- **Suíte de Testes Automatizados (20 testes verdes):**
  - Testes unitários e de integração para loaders, particionamento temporal, engenharia de atributos, previsões do hurdle model e geração gráfica.
  - Suíte de caracterização (*Golden Master*) em `tests/characterization/test_pinning_hurdle_pipeline.py` congelando alvos, splits e invariantes de negócio.
- **Relatório Executivo Quarto em 6 Seções CRISP-ML(Q):**
  - Estruturação nas 6 etapas canônicas: *Contexto de Negócio & Hipóteses*, *Auditoria de Dados & Contratos*, *Análise Exploratória & Atributos*, *Modelagem & Baselines*, *Avaliação & Explicabilidade* e *Recomendações Executivas*.
  - Tabela comparativa formal multi-métrica comparando modelos candidatos contra baselines ingênuos (DummyClassifier e DummyRegressor).
  - 8 figuras de alta resolução (300 DPI) com referências cruzadas nativas `@fig-01` a `@fig-08`.
- **Governança Arquitetural:**
  - 5 Registros de Decisão Arquitetural (ADRs 0001 a 0005) documentando a escolha do Hurdle Model, particionamento temporal, gradient boosting vs deep learning, preservação de N/A e minimização de custo.
  - Navbar reestruturada com dropdowns para acesso direto à arquitetura e ADRs.

### Modificado
- `_quarto.yml`: Configurado `project: render:` explícito, restringindo a compilação pública a arquivos documentais e impedindo a exposição de artefatos de controle interno (`task.md`).
- `pyproject.toml`: Harmonizado target de compatibilidade para `requires-python = ">=3.11"` e adicionado silenciamento seletivo de avisos de depreciação de bibliotecas externas.
- Fidelidade numérica corrigida no relatório: contabilização exata de 1.341 fraudes no treino, 362 na validação e 352 no teste cego (totalizando 2.055 fraudes).

### Corrigido
- Saneamento de link absoluto local `file:///...` em `docs/03-architecture-and-schema.md`.
- Correção de URLs canônicas da navbar do Quarto apontando para o perfil GitHub correto (`talesgomes27`).
- Eliminação de links manuais quebrados `[Figura X](#fig-0X)` em favor da sintaxe nativa `@fig-`.

### Segurança
- Blindagem do `.gitignore` bloqueando binários de modelos serializados (`*.joblib`, `*.pkl`), arquivos de variáveis de ambiente (`.env*`) e caches de build.
