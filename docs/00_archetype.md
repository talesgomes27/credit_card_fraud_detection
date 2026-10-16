---
archetype: A
stack: Python 3.11+ (Conda science / uv, Scikit-Learn, LightGBM, XGBoost, SHAP, Pytest, Ruff) + Quarto 1.9.38
project_name: credit_card_fraud_detection
gerado_por: audit-and-refactoring-pipeline
data: 2026-10-16T00:20:00-03:00
---

# Metadados de Arquétipo do Projeto

- **Arquétipo Selecionado:** Arquétipo A — Data Science, Analytics & ML (Python/Quarto)
- **Metodologia:** Híbrida CRISP-ML(Q) + Data Storytelling em 6 Seções Canônicas
- **Pipeline de Modelagem:** Hurdle Model em 3 Estágios (Binário `is_fraud`, Multiclasse `fraud_type`, Regressão `fraud_amount_lost`)
- **Regime de Validação:** Split temporal *out-of-time* estrito (Treino: Jan-Ago/2026, Validação: Set-Out/2026, Teste: Nov-Dez/2026)
- **Storytelling e Publicação:** Quarto Website em `docs_site/` (GitHub Pages) com 8 figuras em 300 DPI e rastreabilidade 1:1 com as hipóteses de negócio.
- **Auditoria de Ambiente:** Testes automatizados passando (17/17), linter `ruff` sem violações; Quarto CLI em `/usr/lib/rstudio/resources/app/bin/quarto/bin/quarto` (v1.9.38).
