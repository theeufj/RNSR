# Labelled messy-table set

Used by `rnsr eval-tables --dir` to score table extraction on ugly
documents (repeated headers, TOTAL rows, EU/US styles, scans).

## Layout

```
messy-tables/
  labels.json          # required
  *.pdf / *.xlsx / …   # source documents
```

Generate the synthetic set:

```bash
python -c "from rnsr.eval.datasets.messy_tables import generate_messy_tables; generate_messy_tables('testMatter/messy-tables')"
```

## labels.json

```json
{
  "tables": [
    {
      "doc": "bank_statement.pdf",
      "page": 1,
      "table_idx": 0,
      "expected": {
        "headers": ["Date", "Description", "Amount"],
        "n_data_rows": 12,
        "n_total_rows": 1,
        "numeric_columns": ["amount"],
        "style": "us"
      },
      "must_pass": true
    }
  ]
}
```

- `doc` is the filename (not the path).
- `table_idx` is 0-based among tables extracted from that document.
- `must_pass: false` marks bonus cases (e.g. rotated scans) that do not
  fail the suite.
- Drop real bank statements or ledgers here and add a row per table.

## Score

```bash
rnsr eval-tables --dir testMatter/messy-tables
```

Writes `table_score.json` next to the labels.
