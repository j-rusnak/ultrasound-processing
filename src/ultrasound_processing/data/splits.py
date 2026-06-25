from __future__ import annotations

import random
from collections.abc import Iterable


def split_patient_ids(
    patient_ids: Iterable[str | int],
    *,
    train_fraction: float = 0.9,
    seed: int = 42,
    test_patient_ids: Iterable[str | int] | None = None,
) -> tuple[list[str], list[str]]:
    patients = [str(patient_id) for patient_id in patient_ids]
    if test_patient_ids:
        test_set = {str(patient_id) for patient_id in test_patient_ids}
        train = [patient_id for patient_id in patients if patient_id not in test_set]
        test = [patient_id for patient_id in patients if patient_id in test_set]
        return train, test

    shuffled = list(patients)
    random.Random(seed).shuffle(shuffled)
    split = max(1, int(len(shuffled) * train_fraction)) if len(shuffled) > 1 else len(shuffled)
    return shuffled[:split], shuffled[split:]
