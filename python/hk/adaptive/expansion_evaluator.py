"""
HK Adaptive Neural Framework: Expansion Need Evaluator
Diagnoses whether a model has sufficient vocabulary and representational capacity
for a target domain or programming language, and recommends dynamic expansion parameters.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Tuple
import torch
import torch.nn as nn

from .growth import GrowthGovernor


@dataclass
class ExpansionDiagnosis:
    """Diagnostic report describing whether a model needs architectural expansion."""
    needs_expansion: bool
    needs_vocab_expansion: bool
    suggested_new_tokens: List[str] = field(default_factory=list)
    needs_width_expansion: bool = False
    suggested_width_ratio: float = 1.0
    rationale: str = ""
    diagnostics: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"Expansion Diagnosis: {'EXPANSION RECOMMENDED' if self.needs_expansion else 'CAPACITY SUFFICIENT'}",
            f"  - Vocab Expansion: {'YES' if self.needs_vocab_expansion else 'NO'} ({len(self.suggested_new_tokens)} new tokens suggested)",
            f"  - Width Expansion: {'YES' if self.needs_width_expansion else 'NO'} (ratio: {self.suggested_width_ratio:.2f}x)",
            f"  - Rationale: {self.rationale}",
            f"  - Diagnostics: {self.diagnostics}",
        ]
        return "\n".join(lines)


class ExpansionEvaluator:
    """
    Evaluates epistemic capacity and representational bottlenecks for a given domain
    (e.g., learning a new programming language, domain-specific language, or foreign syntax).
    """

    def __init__(
        self,
        governor: Optional[GrowthGovernor] = None,
        max_error_threshold: float = 0.40,
        high_perplexity_threshold: float = 5.0,
        default_width_ratio: float = 1.33,
    ):
        self.governor = governor or GrowthGovernor(max_growth_ratio=2.0)
        self.max_error_threshold = max_error_threshold
        self.high_perplexity_threshold = high_perplexity_threshold
        self.default_width_ratio = default_width_ratio

    def evaluate_vocab_deficit(
        self,
        known_vocab: Set[str],
        domain_keywords: List[str],
    ) -> Tuple[bool, List[str]]:
        """
        Identifies domain keywords or syntax tokens that do not exist in the known vocabulary.
        """
        missing = [kw for kw in domain_keywords if kw not in known_vocab]
        needs_vocab = len(missing) > 0
        return needs_vocab, missing

    def evaluate_domain_capacity(
        self,
        model: nn.Module,
        diagnostic_loss: float,
        diagnostic_pass_rate: float,
        known_vocab: Optional[Set[str]] = None,
        domain_keywords: Optional[List[str]] = None,
        total_vocab_size: Optional[int] = None,
    ) -> ExpansionDiagnosis:
        """
        Synthesizes task error rate, domain loss/perplexity, and vocabulary deficits
        to determine if dynamic expansion is required.
        """
        diagnostics = {
            "diagnostic_loss": diagnostic_loss,
            "diagnostic_pass_rate": diagnostic_pass_rate,
            "diagnostic_perplexity": math.exp(min(diagnostic_loss, 20.0)),
        }

        # 1. Vocab deficit analysis
        needs_vocab = False
        missing_tokens: List[str] = []
        if known_vocab is not None and domain_keywords is not None:
            needs_vocab, missing_tokens = self.evaluate_vocab_deficit(known_vocab, domain_keywords)
            diagnostics["missing_token_count"] = float(len(missing_tokens))

        # 2. Width deficit analysis
        # If pass rate is unacceptably low or perplexity indicates the model cannot capture
        # the domain's syntactic/semantic structures, width expansion is required.
        error_rate = 1.0 - diagnostic_pass_rate
        diagnostics["error_rate"] = error_rate

        needs_width = False
        suggested_ratio = 1.0
        reasons = []

        if needs_vocab:
            reasons.append(f"Domain contains {len(missing_tokens)} unrecognized syntax/keyword tokens.")

        if error_rate > self.max_error_threshold or diagnostic_loss > self.high_perplexity_threshold:
            needs_width = True
            # Propose ratio proportional to deficit, bounded by governor
            if error_rate > 0.70:
                proposed_ratio = 1.50
            elif error_rate > 0.40:
                proposed_ratio = 1.33
            else:
                proposed_ratio = 1.25

            # Consult GrowthGovernor
            current_dim = getattr(getattr(model, "config", None), "intermediate_size", 128)
            approval = self.governor.request_growth(current_dim, int(current_dim * proposed_ratio))
            if approval.approved:
                suggested_ratio = proposed_ratio
                reasons.append(
                    f"High error rate ({error_rate:.1%}) indicates representational capacity bottleneck. "
                    f"Governor approved {suggested_ratio:.2f}x expansion."
                )
            else:
                suggested_ratio = 1.0
                needs_width = False
                reasons.append(f"Capacity expansion was requested but rejected by governor: {approval.reason}")

        needs_expansion = needs_vocab or needs_width
        if not needs_expansion:
            rationale = "Model capacity and vocabulary are sufficient for the target domain."
        else:
            rationale = " ".join(reasons)

        return ExpansionDiagnosis(
            needs_expansion=needs_expansion,
            needs_vocab_expansion=needs_vocab,
            suggested_new_tokens=missing_tokens,
            needs_width_expansion=needs_width,
            suggested_width_ratio=suggested_ratio,
            rationale=rationale,
            diagnostics=diagnostics,
        )
