#!/usr/bin/env python3
"""Ensure the validation project profile is complete for acceptance testing."""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.logging import get_logger
from app.db.supabase import get_supabase_client

logger = get_logger(__name__)

PROJECT_ID = "8dd1894a-8977-468c-b3ae-682d548d400b"
TENANT_ID = "dev-tenant"


def patch_validation_project() -> None:
    db = get_supabase_client()
    result = (
        db.table("projects")
        .select("id, tenant_id, project_profile")
        .eq("id", PROJECT_ID)
        .eq("tenant_id", TENANT_ID)
        .limit(1)
        .execute()
    )
    if not result.data:
        raise RuntimeError(f"Project {PROJECT_ID} not found for tenant {TENANT_ID}")

    row = result.data[0]
    profile = dict(row.get("project_profile") or {})

    data_used = list(profile.get("data_used") or [])
    for tag in ("pii", "pci"):
        if tag not in data_used:
            data_used.append(tag)
    profile["data_used"] = data_used

    if not profile.get("existing_vendor"):
        profile["existing_vendor"] = (
            "Buró de crédito, score alternativo, librería de explicabilidad"
        )

    profile["notes"] = (
        "Equipo RACI: Laura Méndez (Producto), Andrés Salazar (Riesgo), "
        "María Porras (Legal).\n"
        "Comité: Comité de Riesgo de Modelos de Crédito.\n"
        "Archivos del expediente: POL-IA-Credito-Consumo.pdf, "
        "INV-IA-Credito-Consumo.pdf."
    )

    db.table("projects").update(
        {
            "project_profile": profile,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    ).eq("id", PROJECT_ID).eq("tenant_id", TENANT_ID).execute()

    logger.info(
        "Patched validation project %s: data_used=%s, has_notes=%s, has_vendor=%s",
        PROJECT_ID,
        profile.get("data_used"),
        bool(profile.get("notes")),
        bool(profile.get("existing_vendor")),
    )


if __name__ == "__main__":
    patch_validation_project()
