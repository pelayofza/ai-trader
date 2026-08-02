from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_VALIDATION_FRACTION = 0.27  # ~8 de 30 escenarios reservados como hold-out


@dataclass(slots=True, frozen=True)
class ScenarioSplit:
    """
    Particion de ESCENARIOS ENTEROS (no de paths) en train y validation.

    La unidad de hold-out es el escenario/arquetipo macro completo: si un path de
    validacion perteneciera a un escenario presente en train, seria fuga de arquetipo
    y el gap de overfitting quedaria enmascarado. Por eso se parte por scenario_id.
    """

    train: tuple[str, ...]
    validation: tuple[str, ...]
    seed: int

    @property
    def n_train(self) -> int:
        return len(self.train)

    @property
    def n_validation(self) -> int:
        return len(self.validation)


def split_scenarios(
    scenario_ids: list[str],
    *,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    seed: int = 0,
) -> ScenarioSplit:
    """
    Reparte los escenarios en train/validation de forma DETERMINISTA y reproducible.

    Determinismo: se ordenan los ids y se permutan con numpy.default_rng(seed). No se
    usa hash() de Python (esta salado por proceso y romperia la reproducibilidad entre
    ejecuciones). Mismo (ids, seed) -> misma particion, siempre.
    """
    if not scenario_ids:
        raise ValueError("scenario_ids cannot be empty")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")

    unique_sorted = sorted(set(scenario_ids))
    if len(unique_sorted) < 2:
        raise ValueError("need at least 2 distinct scenarios to split")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(unique_sorted))

    n_val = round(len(unique_sorted) * validation_fraction)
    # Garantiza al menos 1 escenario en cada lado.
    n_val = min(max(n_val, 1), len(unique_sorted) - 1)

    val_idx = sorted(order[:n_val])
    train_idx = sorted(order[n_val:])

    validation = tuple(unique_sorted[i] for i in val_idx)
    train = tuple(unique_sorted[i] for i in train_idx)
    return ScenarioSplit(train=train, validation=validation, seed=seed)
