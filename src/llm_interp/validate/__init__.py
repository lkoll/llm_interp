from .base import CheckResult, Tolerance, ValidationContext, get_tolerance
from .checks import (
    check_a_logits_from_final_residual,
    check_b_kv_cache_vs_teacher_forced,
    check_c_recomputed_qk,
    check_d_attention_probs_vs_eager,
    check_e_storage_round_trip,
    check_f_add_vector,
)
from .doctor import main, print_report, run_doctor

__all__ = [
    "CheckResult",
    "Tolerance",
    "ValidationContext",
    "check_a_logits_from_final_residual",
    "check_b_kv_cache_vs_teacher_forced",
    "check_c_recomputed_qk",
    "check_d_attention_probs_vs_eager",
    "check_e_storage_round_trip",
    "check_f_add_vector",
    "get_tolerance",
    "main",
    "print_report",
    "run_doctor",
]
