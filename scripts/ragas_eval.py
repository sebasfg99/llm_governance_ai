# ============================================================
# scripts/ragas_eval.py — Evaluación RAGAs del pipeline RAG
#
# Ejecuta el golden dataset (evals/golden_dataset_cr.jsonl) contra
# el sidecar local (/ai/query, stream=false) y calcula las métricas
# estándar de RAGAs:
#
#   - faithfulness:      ¿la respuesta se sostiene en los contextos?
#   - answer_relevancy:  ¿la respuesta atiende la pregunta?
#   - context_precision: ¿los chunks recuperados son relevantes?
#   - context_recall:    ¿los contextos cubren el ground truth?
#
# Los casos "idk" (fuera de corpus) se EXCLUYEN del agregado RAGAs
# (una negativa correcta puntúa mal en relevancy) y se evalúan por
# separado: se espera que el agente decline responder.
#
# Uso:
#   .venv/bin/python scripts/ragas_eval.py --label baseline
#   .venv/bin/python scripts/ragas_eval.py --label post --base-url http://localhost:8000
#
# Requiere: pip install -r requirements-eval.txt  (ragas, datasets,
# langchain-openai) y OPENAI_API_KEY en el entorno/.env (judge).
#
# Salidas:
#   evals/results/ragas_<label>_<fecha>.json  (métricas + por-caso)
#   tabla markdown por stdout
# ============================================================

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "evals" / "golden_dataset_cr.jsonl"
RESULTS_DIR = ROOT / "evals" / "results"

# Tenant/usuario sintéticos para la evaluación — el sidecar solo los
# usa para scoping y trazas; no se requiere que existan en GovAI.
EVAL_TENANT = "38e504af-e407-44b9-a11c-02e941687488"
EVAL_USER = "ragas-eval"

# Frases que cuentan como "declinó responder" en los casos idk.
_IDK_PATTERNS = re.compile(
    r"no (dispongo|cuento|tengo|encuentro|hay) .{0,40}(informaci[oó]n|datos|documentos)"
    r"|no (he encontrado|encontr[eé]|se encuentra) .{0,40}(informaci[oó]n|datos|en la base|en los documentos|documentos)"
    r"|no est[aá] (disponible|cubiert|inclui)"
    r"|fuera de (mi |la )?(base de conocimiento|alcance)"
    r"|no forma parte de (la|mi) base",
    re.IGNORECASE,
)


def load_dataset() -> list[dict]:
    rows = []
    with open(DATASET, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def query_sidecar(base_url: str, service_key: str, case: dict, top_k: int) -> dict:
    payload = {
        "query": case["question"],
        "use_case": "chatbot",
        "stream": False,
        "top_k": top_k,
        "include_global": True,
        "country": case.get("country"),
        "industry": case.get("industry"),
    }
    req = urllib.request.Request(
        f"{base_url}/ai/query",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {service_key}",
            "X-Tenant-Id": EVAL_TENANT,
            "X-User-Id": EVAL_USER,
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def fetch_full_chunks(chunk_ids: list[str]) -> dict[str, str]:
    """
    retrieved_chunks del API trae content_preview truncado; para
    faithfulness necesitamos el contenido completo. Lo leemos de
    document_chunks vía Supabase REST con la service key del .env.
    """
    supabase_url = os.environ.get("SUPABASE_URL", "").rstrip("/")
    key = os.environ.get("SUPABASE_SERVICE_KEY", "")
    if not supabase_url or not key or not chunk_ids:
        return {}
    ids = ",".join(f'"{c}"' for c in chunk_ids)
    url = f"{supabase_url}/rest/v1/document_chunks?select=id,content&id=in.({ids})"
    req = urllib.request.Request(url, headers={"apikey": key, "Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return {row["id"]: row["content"] for row in json.load(r)}
    except Exception as e:  # noqa: BLE001 — eval tooling, degradar a preview
        print(f"  ⚠ no se pudo leer chunks completos ({e}); uso previews", file=sys.stderr)
        return {}


def run_cases(base_url: str, service_key: str, cases: list[dict], top_k: int) -> list[dict]:
    out = []
    for case in cases:
        t0 = time.time()
        try:
            resp = query_sidecar(base_url, service_key, case, top_k)
        except Exception as e:  # noqa: BLE001
            print(f"[ERR ] {case['id']}: {e}", file=sys.stderr)
            out.append({**case, "answer": f"[ERROR: {e}]", "contexts": [], "latency_s": time.time() - t0})
            continue
        chunk_ids = [c["chunk_id"] for c in resp.get("retrieved_chunks", [])]
        full = fetch_full_chunks(chunk_ids)
        contexts = [
            full.get(c["chunk_id"], c.get("content_preview", ""))
            for c in resp.get("retrieved_chunks", [])
        ]
        out.append({
            **case,
            "answer": resp.get("response_text", ""),
            "contexts": contexts,
            "model": f"{resp.get('model_provider')}/{resp.get('model_name')}",
            "latency_s": round(time.time() - t0, 1),
            "cost_usd": resp.get("cost_estimate_usd"),
            "trace_id": resp.get("trace_id"),
        })
        print(f"[ok  ] {case['id']} ({out[-1]['latency_s']}s, {len(contexts)} chunks)", file=sys.stderr)
    return out


def eval_idk(rows: list[dict]) -> dict:
    results = {}
    for r in rows:
        declined = bool(_IDK_PATTERNS.search(r["answer"]))
        results[r["id"]] = {"declined": declined, "answer_head": r["answer"][:160]}
    total = len(results)
    correct = sum(1 for v in results.values() if v["declined"])
    return {"accuracy": round(correct / total, 3) if total else None, "total": total,
            "correct": correct, "cases": results}


def eval_ragas(rows: list[dict]) -> dict:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    ds = Dataset.from_dict({
        "question": [r["question"] for r in rows],
        "answer": [r["answer"] for r in rows],
        "contexts": [r["contexts"] for r in rows],
        "ground_truth": [r["ground_truth"] for r in rows],
    })

    judge = None
    try:
        from langchain_openai import ChatOpenAI
        from ragas.llms import LangchainLLMWrapper
        judge = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0))
    except Exception:  # noqa: BLE001 — usar judge por defecto de ragas
        pass

    kwargs = {"llm": judge} if judge else {}
    scores = evaluate(ds, metrics=[faithfulness, answer_relevancy, context_precision, context_recall], **kwargs)
    df = scores.to_pandas()

    metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    aggregate = {m: round(float(df[m].mean()), 4) for m in metric_cols if m in df}
    per_case = []
    for i, r in enumerate(rows):
        per_case.append({"id": r["id"], **{m: (round(float(df[m][i]), 4) if df[m][i] == df[m][i] else None) for m in metric_cols if m in df}})
    return {"aggregate": aggregate, "per_case": per_case}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluación RAGAs del sidecar GovAI")
    parser.add_argument("--label", default="baseline", help="Etiqueta del run (baseline|post|...)")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--skip-ragas", action="store_true",
                        help="Solo ejecutar queries y guardar respuestas (sin métricas)")
    args = parser.parse_args()

    # .env del sidecar: OPENAI_API_KEY (judge), SUPABASE_* (chunks completos)
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass

    service_key = os.environ.get("AI_CORE_SERVICE_KEY", "")
    if not service_key:
        sys.exit("AI_CORE_SERVICE_KEY no configurada (revisa llm_governance_ai/.env)")

    cases = load_dataset()
    answerable = [c for c in cases if c["case_type"] == "answerable"]
    idk = [c for c in cases if c["case_type"] == "idk"]
    print(f"Dataset: {len(answerable)} respondibles + {len(idk)} idk", file=sys.stderr)

    rows_answerable = run_cases(args.base_url, service_key, answerable, args.top_k)
    rows_idk = run_cases(args.base_url, service_key, idk, args.top_k)

    result = {
        "label": args.label,
        "date": date.today().isoformat(),
        "base_url": args.base_url,
        "top_k": args.top_k,
        "model": next((r.get("model") for r in rows_answerable if r.get("model")), None),
        "idk": eval_idk(rows_idk),
        "responses": {r["id"]: {"answer": r["answer"], "latency_s": r.get("latency_s"),
                                "cost_usd": r.get("cost_usd"), "n_contexts": len(r["contexts"])}
                      for r in rows_answerable + rows_idk},
    }

    if not args.skip_ragas:
        valid = [r for r in rows_answerable if r["contexts"] and not r["answer"].startswith("[ERROR")]
        print(f"Ejecutando RAGAs sobre {len(valid)} casos…", file=sys.stderr)
        result["ragas"] = eval_ragas(valid)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"ragas_{args.label}_{date.today().isoformat()}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nResultados → {out_path}", file=sys.stderr)

    # Tabla resumen
    print(f"\n## RAGAs — {args.label} ({result['date']}) · modelo {result.get('model')}\n")
    if "ragas" in result:
        print("| Métrica | Score |")
        print("|---|---|")
        for m, v in result["ragas"]["aggregate"].items():
            print(f"| {m} | {v} |")
    print(f"| idk_accuracy | {result['idk']['accuracy']} ({result['idk']['correct']}/{result['idk']['total']}) |")


if __name__ == "__main__":
    main()
