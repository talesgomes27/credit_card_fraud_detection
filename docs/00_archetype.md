---
archetype: A
stack: Python 3.11+ (uv, Scikit-Learn, LightGBM, XGBoost, SHAP, Pytest, Ruff) + Quarto
project_name: credit_card_fraud_detection
gerado_por: pre-code-pipeline
data: 2026-10-14T11:00:00-03:00
---

# Metadados de Arquétipo do Projeto

- **Arquétipo Selecionado:** Arquétipo A — Data Science, Analytics & ML (Python/Quarto)
- **Metodologia:** Híbrida CRISP-ML(Q) + Six Phases of Data Analytics (`Ask`, `Prepare`, `Process`, `Construct`, `Share`, `Act`)
- **Pipeline de Modelagem:** Hurdle Model em 3 Estágios (Binário `is_fraud`, Multiclasse `fraud_type`, Regressão `fraud_amount_lost`)
- **Regime de Validação:** Split temporal *out-of-time* estrito (Treino: Jan-Ago/2026, Validação: Set-Out/2026, Teste: Nov-Dez/2026)
- **Storytelling e Publicação:** Quarto Website em `docs_site/` (GitHub Pages) com 8 figuras em 300 DPI e rastreabilidade 1:1 com as hipóteses de negócio.
