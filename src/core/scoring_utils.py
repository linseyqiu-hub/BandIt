from core.config import VALID_OVERALL_SCORES


def validate_overall_score(value):
    """
    Returns (ok: bool, normalized_value: float or None).
    Snaps to the nearest 0.5 and checks that snap was actually close to the
    original value, so e.g. 6.3 correctly fails rather than rounding to 6.5.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False, None
    snapped = round(v * 2) / 2
    if abs(snapped - v) > 1e-6:
        return False, None
    if snapped not in VALID_OVERALL_SCORES:
        return False, None
    return True, snapped


def compute_band_bin(overall: float) -> str:
    if overall < 5.0:
        return "poor"
    elif overall < 6.5:
        return "developing"
    elif overall < 8.0:
        return "competent"
    else:
        return "expert"
