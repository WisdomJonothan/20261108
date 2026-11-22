cd RETECO
ls -la data/track1_tempo/bitcoin/ | head -12
grep -c "d90887dc_0002" data/track1_tempo/bitcoin/documents.jsonl
grep -o "\"id\": \"[^\"]*\"" data/track1_tempo/bitcoin/documents.jsonl | head -3
grep -rn "documents.jsonl\|corpus.jsonl" scripts/followir_dense.py scripts/dense_qwen3.py 2>/dev/null | head
