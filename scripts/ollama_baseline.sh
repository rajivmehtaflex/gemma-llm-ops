#!/usr/bin/env bash
# Block #1: frozen prompts -> Ollama -> reports JSONL
set -euo pipefail
MODEL="${MODEL:-gemma3:4b}"
PROMPTS="${PROMPTS:-data/baseline/shell_prompts.jsonl}"
OUT="${OUT:-data/baseline/baseline_reports.jsonl}"
SEED="${SEED:-42}"; NUM_PREDICT="${NUM_PREDICT:-512}"
mkdir -p "$(dirname "$OUT")"; : > "$OUT"

while IFS= read -r line; do
  id=$(jq -r '.id' <<<"$line"); prompt=$(jq -r '.prompt' <<<"$line")
  echo ">> $id"
  body=$(jq -n --arg m "$MODEL" --arg p "$prompt" --argjson s "$SEED" --argjson n "$NUM_PREDICT" \
    '{model:$m, stream:false, messages:[{role:"user",content:$p}],
      options:{temperature:0.7, top_p:0.9, seed:$s, num_predict:$n}}')
  resp=$(curl -sS --max-time 300 -w $'\n%{time_total}' localhost:11434/api/chat -d "$body")
  t=$(tail -n1 <<<"$resp"); content=$(head -n -1 <<<"$resp" | jq -r '.message.content')
  jq -n --arg id "$id" --arg p "$prompt" --arg c "$content" --arg m "$MODEL" \
    --arg s "$SEED" --arg n "$NUM_PREDICT" --arg t "$t" \
    '{id:$id, prompt:$p, response:$c,
      settings:{model:$m, temperature:0.7, top_p:0.9, seed:($s|tonumber), num_predict:($n|tonumber)},
      latency_sec:($t|tonumber), notes:""}'
done < "$PROMPTS" >> "$OUT"
echo "wrote $(wc -l < "$OUT") reports -> $OUT"
