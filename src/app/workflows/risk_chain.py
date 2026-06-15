from __future__ import annotations

from typing import Any

from app.assessment.schemas import AssessmentFinding, FoundationAssessmentPacket


def build_risk_assessment_chains(
    *,
    packet: FoundationAssessmentPacket,
    weaknesses: list[AssessmentFinding],
    risk_evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    chains = [
        _risk_chain(packet, finding, risk_evidence[index] if index < len(risk_evidence) else {})
        for index, finding in enumerate(weaknesses)
    ]
    return {
        "risk_assessment_chains": chains,
        "toolchain_delta": _toolchain_delta(chains),
    }


def _risk_chain(
    packet: FoundationAssessmentPacket,
    finding: AssessmentFinding,
    evidence_package: dict[str, Any],
) -> dict[str, Any]:
    result = next(
        (
            item
            for item in packet.questionnaire_results
            if item.question_id == finding.question_id
        ),
        None,
    )
    answer = evidence_package.get("answer") or {}
    matrix = answer.get("risk_control_matrix") or []
    controls = _controls_by_function(answer.get("recommended_controls") or [])
    known = [
        f"{finding.control.framework} {finding.control.control_id} ({finding.control.title}) "
        f"is {finding.compliance.value} compliance with {finding.maturity.value} maturity.",
        finding.summary,
    ]
    if result and result.vendor_comment:
        known.append(f"Vendor comment: {result.vendor_comment}")
    if result and result.analyst_comment:
        known.append(f"Analyst comment: {result.analyst_comment}")
    standards_requirements = [
        {
            "control": control,
            "function": _control_function(control),
            "evidence": _source_ids_for_control(control, evidence_package.get("sources") or []),
        }
        for control in answer.get("recommended_controls") or []
    ]
    inherent = _inherent_risk(packet, finding, answer, matrix)
    control_effects = _control_effects(controls)
    resilience = _resilience_effects(controls, answer, finding)
    return {
        "question_id": finding.question_id,
        "linked_control": finding.control.model_dump(mode="json"),
        "tier_context": {
            "level": packet.tier.level,
            "definition": packet.tier.definition,
            "materiality": _tier_materiality(packet),
        },
        "known_from_assessment": _dedupe(known),
        "standards_requirements_added": standards_requirements,
        "confirmed_gaps": _dedupe([row.get("gap") for row in matrix] or [finding.summary]),
        "threat_scenarios": _dedupe(answer.get("threats") or []),
        "vulnerabilities": _dedupe(answer.get("vulnerabilities") or []),
        "inherent_risk": inherent,
        "recommended_controls_by_function": controls,
        "control_effects": control_effects,
        "resilience_effects": resilience,
        "residual_concern": _residual_concern(inherent, controls, resilience),
        "missing_information": _dedupe(answer.get("missing_information") or []),
        "source_mappings": evidence_package.get("sources") or [],
        "added_value_summary": _added_value_summary(
            answer=answer,
            controls=controls,
            resilience=resilience,
        ),
    }


def _toolchain_delta(chains: list[dict[str, Any]]) -> dict[str, Any]:
    added_controls: list[str] = []
    added_risks: list[str] = []
    resilience_findings: list[str] = []
    missing: list[str] = []
    graph_added: list[str] = []
    for chain in chains:
        for requirement in chain.get("standards_requirements_added") or []:
            control = requirement.get("control")
            if control:
                added_controls.append(control)
        risk = chain.get("inherent_risk") or {}
        if risk.get("risk_statement"):
            added_risks.append(risk["risk_statement"])
        for item in chain.get("resilience_effects") or []:
            resilience_findings.append(item)
        missing.extend(chain.get("missing_information") or [])
    return {
        "already_known_from_sql": [
            item
            for chain in chains
            for item in chain.get("known_from_assessment", [])[:2]
        ],
        "added_by_rag": _dedupe([*added_controls, *added_risks]),
        "added_by_graphrag": graph_added,
        "added_by_resilience_analysis": _dedupe(resilience_findings),
        "remaining_uncertainty": _dedupe(missing),
        "business_interpretation": (
            "The questionnaire identifies weak controls; the toolchain maps those weaknesses "
            "to standards-backed controls, risk mechanisms, resilience implications, and "
            "missing evidence needed for analyst review."
        ),
    }


def _controls_by_function(controls: list[str]) -> dict[str, list[str]]:
    grouped = {
        "preventative": [],
        "detective": [],
        "corrective": [],
        "recovery": [],
        "response": [],
        "governance": [],
    }
    for control in controls:
        grouped[_control_function(control)].append(control)
    return {key: _dedupe(value) for key, value in grouped.items() if value}


def _control_function(control: str) -> str:
    lower = control.lower()
    if any(term in lower for term in ["recovery", "restore", "continuity", "bcd"]):
        return "recovery"
    if any(term in lower for term in ["communication", "incident response", "log"]):
        return "response"
    if any(term in lower for term in ["detect", "monitor", "malicious code", "malware"]):
        return "detective"
    if any(term in lower for term in ["coordinate", "provider", "governance"]):
        return "governance"
    if any(term in lower for term in ["deploy", "protect", "prevent", "always on"]):
        return "preventative"
    return "corrective"


def _source_ids_for_control(control: str, sources: list[dict[str, Any]]) -> list[str]:
    control_lower = control.lower()
    matches = []
    for source in sources:
        metadata = source.get("metadata") or {}
        source_text = " ".join(
            str(value or "")
            for value in [
                metadata.get("control_id"),
                metadata.get("framework"),
                source.get("source"),
            ]
        ).lower()
        if any(token in source_text for token in _control_tokens(control_lower)):
            matches.append(str(source.get("id") or ""))
    return _dedupe(matches)


def _control_tokens(control: str) -> list[str]:
    return [
        token
        for token in control.replace("-", " ").replace(".", " ").split()
        if len(token) >= 3
    ]


def _inherent_risk(
    packet: FoundationAssessmentPacket,
    finding: AssessmentFinding,
    answer: dict[str, Any],
    matrix: list[dict[str, Any]],
) -> dict[str, str]:
    row = matrix[0] if matrix else {}
    likelihood = str(row.get("likelihood") or _default_likelihood(finding))
    impact = str(row.get("impact") or _default_impact(packet))
    risk = str(row.get("risk") or (answer.get("risks") or [""])[0])
    return {
        "likelihood": likelihood,
        "impact": impact,
        "risk_statement": risk or finding.summary,
        "reason": (
            f"Tier {packet.tier.level} vendor context combined with "
            f"{finding.compliance.value} compliance and {finding.maturity.value} maturity."
        ),
    }


def _default_likelihood(finding: AssessmentFinding) -> str:
    if finding.compliance.value == "no" or finding.maturity.value == "basic":
        return "high"
    if finding.compliance.value == "partial":
        return "medium"
    return "unknown"


def _default_impact(packet: FoundationAssessmentPacket) -> str:
    if packet.tier.level <= 2:
        return "high"
    if packet.tier.level == 3:
        return "medium"
    return "low"


def _control_effects(controls: dict[str, list[str]]) -> list[str]:
    effects: list[str] = []
    if controls.get("preventative"):
        effects.append("Preventative controls reduce likelihood when implemented and enforced.")
    if controls.get("detective"):
        effects.append("Detective controls improve detection and support response decisions.")
    if controls.get("corrective"):
        effects.append("Corrective controls reduce recurrence after remediation.")
    if controls.get("response"):
        effects.append("Response controls improve containment and coordination during incidents.")
    if controls.get("recovery"):
        effects.append("Recovery controls reduce impact by restoring operations after disruption.")
    if controls.get("governance"):
        effects.append("Governance controls clarify responsibility and evidence expectations.")
    return effects


def _resilience_effects(
    controls: dict[str, list[str]],
    answer: dict[str, Any],
    finding: AssessmentFinding,
) -> list[str]:
    effects: list[str] = []
    if controls.get("recovery") or "recovery" in finding.control.control_type:
        effects.append(
            "Recovery and continuity controls contribute directly to resilience because they "
            "show whether the vendor can restore operations after an incident."
        )
    if controls.get("response"):
        effects.append(
            "Response controls contribute to resilience by improving incident coordination and "
            "communication."
        )
    if controls.get("detective") and not controls.get("response"):
        effects.append(
            "Detection improves resilience only if response ownership and procedures are also "
            "evidenced."
        )
    if not effects and answer.get("missing_information"):
        effects.append("Resilience contribution cannot be confirmed from the available evidence.")
    return effects


def _residual_concern(
    inherent: dict[str, str],
    controls: dict[str, list[str]],
    resilience: list[str],
) -> dict[str, str]:
    if controls:
        statement = (
            "Residual risk depends on implementation evidence, operating effectiveness, and "
            "whether response/recovery capabilities are tested."
        )
    else:
        statement = "Residual risk cannot be reduced until suitable controls are identified."
    if not resilience:
        statement += " Resilience remains unproven."
    return {
        "expected_after_controls": "lower likelihood or impact if controls are implemented",
        "remaining_issue": statement,
        "starting_likelihood": inherent.get("likelihood", "unknown"),
        "starting_impact": inherent.get("impact", "unknown"),
    }


def _added_value_summary(
    *,
    answer: dict[str, Any],
    controls: dict[str, list[str]],
    resilience: list[str],
) -> str:
    control_count = sum(len(items) for items in controls.values())
    risk_count = len(answer.get("risks") or [])
    parts = [
        f"Added {control_count} standards-backed control mapping(s)",
        f"{risk_count} risk interpretation(s)",
    ]
    if resilience:
        parts.append("resilience implication(s)")
    return ", ".join(parts) + "."


def _tier_materiality(packet: FoundationAssessmentPacket) -> str:
    sensitive = any(
        attribute.name == "sensitive_data_access" and bool(attribute.value)
        for attribute in packet.tier.attributes
    )
    if packet.tier.level <= 2 and sensitive:
        return "material vendor risk because Tier 2 includes sensitive data access"
    if packet.tier.level <= 2:
        return "material vendor risk because of the tier level"
    return "standard vendor risk"


def _dedupe(values: list[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text.lower() in seen:
            continue
        seen.add(text.lower())
        result.append(text)
    return result
