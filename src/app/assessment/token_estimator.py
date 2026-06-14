from __future__ import annotations

from dataclasses import dataclass

from app.assessment.findings import classify_findings, sanitize_packet
from app.assessment.prompts import build_foundation_summary_prompt
from app.assessment.schemas import FoundationAssessmentPacket

MODEL_PRICES_PER_MILLION = {
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}


@dataclass(frozen=True)
class TokenEstimate:
    model: str
    estimated_input_tokens: int
    estimated_output_tokens: int
    estimated_total_tokens: int
    estimated_cost_usd: float
    prompt_characters: int

    def as_dict(self) -> dict[str, int | float | str]:
        return {
            "model": self.model,
            "estimated_input_tokens": self.estimated_input_tokens,
            "estimated_output_tokens": self.estimated_output_tokens,
            "estimated_total_tokens": self.estimated_total_tokens,
            "estimated_cost_usd": self.estimated_cost_usd,
            "prompt_characters": self.prompt_characters,
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
    price = MODEL_PRICES_PER_MILLION.get(model, MODEL_PRICES_PER_MILLION["gpt-4.1-mini"])
    input_cost = input_tokens * price["input"] / 1_000_000
    output_cost = estimated_output_tokens * price["output"] / 1_000_000
    return TokenEstimate(
        model=model,
        estimated_input_tokens=input_tokens,
        estimated_output_tokens=estimated_output_tokens,
        estimated_total_tokens=input_tokens + estimated_output_tokens,
        estimated_cost_usd=round(input_cost + output_cost, 6),
        prompt_characters=len(prompt_text),
    )


def estimate_tokens(text: str) -> int:
    # Conservative rough estimate for English prose and JSON without adding a tokenizer dependency.
    return max(1, (len(text) + 2) // 3)

