from __future__ import annotations

from dataclasses import dataclass

from app.assessment.findings import classify_findings, sanitize_packet
from app.assessment.prompts import build_foundation_summary_prompt
from app.assessment.schemas import FoundationAssessmentPacket

MODEL_PRICES_PER_MILLION = {
    "gpt-5.5": {"input": 5.00, "output": 30.00},
    "gpt-5.4": {"input": 2.50, "output": 15.00},
    "gpt-5.4-mini": {"input": 0.75, "output": 4.50},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
}
UNKNOWN_OPENAI_PRICE_PER_MILLION = {"input": 10.00, "output": 60.00}

USD_TO_EUR_RATE = 0.92


@dataclass(frozen=True)
class TokenEstimate:
    model: str
    estimated_input_tokens: int
    estimated_output_tokens: int
    estimated_total_tokens: int
    estimated_cost_usd: float
    estimated_cost_eur: float
    usd_to_eur_rate: float
    prompt_characters: int
    pricing_note: str

    def as_dict(self) -> dict[str, int | float | str]:
        return {
            "model": self.model,
            "estimated_input_tokens": self.estimated_input_tokens,
            "estimated_output_tokens": self.estimated_output_tokens,
            "estimated_total_tokens": self.estimated_total_tokens,
            "estimated_cost_usd": self.estimated_cost_usd,
            "estimated_cost_eur": self.estimated_cost_eur,
            "usd_to_eur_rate": self.usd_to_eur_rate,
            "prompt_characters": self.prompt_characters,
            "pricing_note": self.pricing_note,
        }


def estimate_foundation_summary_tokens(
    packet: FoundationAssessmentPacket,
    *,
    model: str = "gpt-4.1-mini",
    estimated_output_tokens: int = 900,
) -> TokenEstimate:
    sanitized = sanitize_packet(packet)
    findings = classify_findings(sanitized)
    prompt = build_foundation_summary_prompt(
        sanitized,
        {key: [item.model_dump() for item in value] for key, value in findings.items()},
    )
    prompt_text = "\n".join(message["content"] for message in prompt)
    input_tokens = estimate_tokens(prompt_text)
    price = MODEL_PRICES_PER_MILLION.get(model)
    if price is None and model.startswith("gpt-"):
        price = UNKNOWN_OPENAI_PRICE_PER_MILLION
        pricing_note = (
            "OpenAI model pricing is not configured. Using conservative placeholder pricing "
            "of $10/M input and $60/M output tokens."
        )
        input_cost = input_tokens * price["input"] / 1_000_000
        output_cost = estimated_output_tokens * price["output"] / 1_000_000
    elif price is None:
        input_cost = 0.0
        output_cost = 0.0
        pricing_note = "No API token price configured; local or unknown model is estimated as $0."
    else:
        input_cost = input_tokens * price["input"] / 1_000_000
        output_cost = estimated_output_tokens * price["output"] / 1_000_000
        pricing_note = "Estimated from configured per-million input/output token prices."
    estimated_cost_usd = round(input_cost + output_cost, 6)
    return TokenEstimate(
        model=model,
        estimated_input_tokens=input_tokens,
        estimated_output_tokens=estimated_output_tokens,
        estimated_total_tokens=input_tokens + estimated_output_tokens,
        estimated_cost_usd=estimated_cost_usd,
        estimated_cost_eur=round(estimated_cost_usd * USD_TO_EUR_RATE, 6),
        usd_to_eur_rate=USD_TO_EUR_RATE,
        prompt_characters=len(prompt_text),
        pricing_note=pricing_note,
    )


def estimate_tokens(text: str) -> int:
    # Conservative rough estimate for English prose and JSON without adding a tokenizer dependency.
    return max(1, (len(text) + 2) // 3)
