# ADR 0004: Preservação de 'N/A' na Coluna Browser e Contrato de Leitura

- **Status:** Aceito
- **Data:** 2026-10-14
- **Contexto:** A leitura padrão via `pd.read_csv` converte a string `'N/A'` automaticamente para `np.nan`, afetando 41.709 transações. A auditoria do Gate 1 comprovou que essas 41.709 ocorrências pertencem exclusivamente a transações físicas presenciais (*In-store*: 35.777) e caixas eletrônicos (*ATM*: 5.932). Nesses canais, não existe navegador web. O valor `'N/A'` não representa um dado faltante acidental, mas uma categoria estrutural válida do domínio de negócio.

## Decisão
Estabelecer no contrato de ingestão que o arquivo CSV deve ser lido explicitamente com `keep_default_na=False` ou mapeamento customizado `na_values=['']`.
Na camada de engenharia de features, a coluna `browser` será mantida como string com 7 categorias legítimas:
`['App', 'Chrome', 'Edge', 'Firefox', 'N/A', 'Other', 'Safari']`.

## Alternativas Rejeitadas
- **Imputação por Moda ou Categoria Mais Frequente:** Substituiria `'N/A'` por `'Chrome'` (moda entre compras online), gerando um erro absurdo de domínio: compras físicas no caixa de supermercado apareceriam como feitas via navegador Chrome.
- **Remoção de Linhas com 'N/A':** Descartaria 41,7% da base de dados, eliminando todas as transações presenciais e inviabilizando a análise de proteção por Chip EMV.

## Consequências
- **Positivas:** Preserva a integridade do domínio financeiro, elimina falsos alertas de ausência de dados nas ferramentas de profiling e permite que os algoritmos diferenciem operações web de operações em terminais físicos.
- **Mitigadas:** Qualquer novo desenvolvedor ou script que carregue o CSV sem essa flag receberá 41k NaNs. Para mitigar isso, encapsularemos a leitura em uma função padronizada `load_clean_data()` no módulo `src/data/loader.py` com validação de contrato.
