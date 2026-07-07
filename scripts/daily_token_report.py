# ============================================================
# scripts/daily_token_report.py — Reporte diario de tokens/costos
#
# Agrega el consumo LLM por día × modelo × proveedor desde la
# tabla `traces` (fuente primaria) y añade un desglose UI por
# interaction_type desde `agent_interactions` (fuente secundaria).
#
# REGLA DE DEDUPLICACIÓN (importante):
#   Cada llamada LLM del sidecar genera UNA fila en `traces` (con
#   tokens y costo) y, cuando viene del chat de la UI, TAMBIÉN una
#   fila en `agent_interactions` (sin model_name y usualmente sin
#   tokens). Son vistas de la MISMA llamada:
#     - `traces` es la única fuente de verdad para tokens y costo.
#     - `agent_interactions` solo aporta el desglose por
#       interaction_type y NUNCA suma al total (evita doble conteo).
#
# COSTOS: se prefiere el `cost_estimate_usd` persistido en la traza
# (calculado con la versión de pricing vigente al momento de la
# llamada); solo se recalcula con estimate_cost() cuando falta y hay
# tokens. Todos los montos son ESTIMADOS, no facturación real.
#
# USO:
#   .venv/bin/python scripts/daily_token_report.py
#   .venv/bin/python scripts/daily_token_report.py --since 2026-07-01 --until 2026-07-07
#   .venv/bin/python scripts/daily_token_report.py --client-id <uuid> --format csv
# ============================================================

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.domain.observability.cost import PRICING_VERSION, estimate_cost  # noqa: E402

PAGE_SIZE = 1000


def _env() -> tuple[str, str]:
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    url = os.environ.get("SUPABASE_URL", "").rstrip("/")
    key = os.environ.get("SUPABASE_SERVICE_KEY", "")
    if not url or not key:
        sys.exit("SUPABASE_URL / SUPABASE_SERVICE_KEY no configuradas (revisa .env)")
    return url, key


def _fetch_all(base: str, key: str, table: str, params: dict[str, str]) -> list[dict]:
    """Paginación por rangos sobre PostgREST."""
    rows: list[dict] = []
    offset = 0
    while True:
        req = urllib.request.Request(
            f"{base}/rest/v1/{table}?{urllib.parse.urlencode(params)}",
            headers={
                "apikey": key,
                "Authorization": f"Bearer {key}",
                "Range-Unit": "items",
                "Range": f"{offset}-{offset + PAGE_SIZE - 1}",
            },
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            page = json.load(r)
        rows.extend(page)
        if len(page) < PAGE_SIZE:
            return rows
        offset += PAGE_SIZE


def build_report(since: str, until: str, client_id: str | None) -> dict:
    base, key = _env()

    # ---- Fuente primaria: traces ----
    params = {
        "select": "created_at,model_name,model_provider,prompt_tokens,completion_tokens,cost_estimate_usd,use_case,tenant_id,client_id",
        "created_at": f"gte.{since}",
        "and": f"(created_at.lt.{until})",
        "order": "created_at",
    }
    if client_id:
        params["or"] = f"(tenant_id.eq.{client_id},client_id.eq.{client_id})"
    traces = _fetch_all(base, key, "traces", params)

    daily: dict[tuple[str, str, str], dict] = defaultdict(
        lambda: {"input": 0, "output": 0, "calls": 0, "cost": 0.0, "recalculated": 0}
    )
    for t in traces:
        day = (t["created_at"] or "")[:10]
        model = t.get("model_name") or "desconocido"
        provider = t.get("model_provider") or "desconocido"
        pt = t.get("prompt_tokens") or 0
        ct = t.get("completion_tokens") or 0
        cost = t.get("cost_estimate_usd")
        row = daily[(day, model, provider)]
        if cost is None and (pt or ct):
            cost = estimate_cost(provider=provider, model=model, prompt_tokens=pt, completion_tokens=ct)
            row["recalculated"] += 1
        row["input"] += pt
        row["output"] += ct
        row["calls"] += 1
        row["cost"] += float(cost or 0.0)

    # ---- Fuente secundaria: agent_interactions (solo desglose UI) ----
    ai_params = {
        "select": "created_at,interaction_type",
        "created_at": f"gte.{since}",
        "and": f"(created_at.lt.{until})",
        "order": "created_at",
    }
    if client_id:
        ai_params["institution_id"] = f"eq.{client_id}"
    try:
        interactions = _fetch_all(base, key, "agent_interactions", ai_params)
    except Exception as e:  # noqa: BLE001 — tabla de otro proyecto; degradar
        print(f"⚠ agent_interactions no disponible ({e})", file=sys.stderr)
        interactions = []

    ui_daily: dict[tuple[str, str], int] = defaultdict(int)
    for i in interactions:
        ui_daily[((i["created_at"] or "")[:10], i.get("interaction_type") or "desconocido")] += 1

    return {"daily": daily, "ui_daily": ui_daily, "since": since, "until": until,
            "client_id": client_id, "n_traces": len(traces), "n_interactions": len(interactions)}


def render_md(r: dict) -> str:
    out = io.StringIO()
    out.write(f"# Reporte de tokens y costos — {r['since']} a {r['until']}\n\n")
    if r["client_id"]:
        out.write(f"Cliente/institución: `{r['client_id']}`\n\n")
    out.write(f"Fuente primaria: `traces` ({r['n_traces']} llamadas). "
              f"Costos **estimados** (pricing {PRICING_VERSION}), no facturación real.\n\n")

    out.write("| día | modelo | proveedor | input_tokens | output_tokens | total_tokens | llamadas | costo_usd_aprox |\n")
    out.write("|---|---|---|---|---|---|---|---|\n")
    tot_in = tot_out = tot_calls = 0
    tot_cost = 0.0
    for (day, model, provider), v in sorted(r["daily"].items()):
        out.write(f"| {day} | {model} | {provider} | {v['input']:,} | {v['output']:,} "
                  f"| {v['input'] + v['output']:,} | {v['calls']} | ${v['cost']:.4f} |\n")
        tot_in += v["input"]; tot_out += v["output"]; tot_calls += v["calls"]; tot_cost += v["cost"]
    out.write(f"| **total** | | | **{tot_in:,}** | **{tot_out:,}** | **{tot_in + tot_out:,}** "
              f"| **{tot_calls}** | **${tot_cost:.4f}** |\n\n")

    if r["ui_daily"]:
        out.write("## Desglose UI por interaction_type (informativo)\n\n")
        out.write("Vista de `agent_interactions` — son las MISMAS llamadas ya contadas "
                  "arriba cuando provienen del chat; no se suman al total.\n\n")
        out.write("| día | interaction_type | interacciones |\n|---|---|---|\n")
        for (day, itype), n in sorted(r["ui_daily"].items()):
            out.write(f"| {day} | {itype} | {n} |\n")
    return out.getvalue()


def render_csv(r: dict) -> str:
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["day", "model", "provider", "input_tokens", "output_tokens",
                "total_tokens", "interactions", "cost_usd_approx", "pricing_version"])
    for (day, model, provider), v in sorted(r["daily"].items()):
        w.writerow([day, model, provider, v["input"], v["output"],
                    v["input"] + v["output"], v["calls"], round(v["cost"], 6), PRICING_VERSION])
    return out.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description="Reporte diario de tokens/costos LLM")
    today = date.today()
    parser.add_argument("--since", default=(today - timedelta(days=7)).isoformat(),
                        help="Fecha inicial YYYY-MM-DD (default: hace 7 días)")
    parser.add_argument("--until", default=(today + timedelta(days=1)).isoformat(),
                        help="Fecha final exclusiva YYYY-MM-DD (default: mañana)")
    parser.add_argument("--client-id", default=None, help="Filtrar por tenant/institución (uuid)")
    parser.add_argument("--format", choices=["md", "csv"], default="md")
    args = parser.parse_args()

    datetime.fromisoformat(args.since)
    datetime.fromisoformat(args.until)

    report = build_report(args.since, args.until, args.client_id)
    print(render_md(report) if args.format == "md" else render_csv(report))


if __name__ == "__main__":
    main()
