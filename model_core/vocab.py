from dataclasses import dataclass

from .ops import OPS_CONFIG

# BACKWARD COMPAT: FEATURE_NAMES order is frozen (ids 0-5). New features MUST be
# APPENDED via ADVANCED_FEATURE_NAMES; never reorder existing entries.
FEATURE_NAMES = (
    "RET",
    "LIQ_SCORE",
    "PRESSURE",
    "FOMO",
    "DEV",
    "LOG_VOL",
)

ADVANCED_FEATURE_NAMES = FEATURE_NAMES + (
    "VOL_CLUSTER",
    "MOM_REV",
    "REL_STRENGTH",
    "HL_RANGE",
    "CLOSE_POS",
    "VOL_TREND",
)


@dataclass(frozen=True)
class FormulaVocab:
    feature_names: tuple[str, ...]
    operator_names: tuple[str, ...]

    @property
    def feature_count(self) -> int:
        return len(self.feature_names)

    @property
    def operator_offset(self) -> int:
        return self.feature_count

    @property
    def token_names(self) -> tuple[str, ...]:
        return self.feature_names + self.operator_names

    @property
    def size(self) -> int:
        return len(self.token_names)


def get_vocab(use_advanced: bool = False) -> FormulaVocab:
    names = ADVANCED_FEATURE_NAMES if use_advanced else FEATURE_NAMES
    return FormulaVocab(feature_names=names, operator_names=tuple(cfg[0] for cfg in OPS_CONFIG))


FORMULA_VOCAB = FormulaVocab(
    feature_names=FEATURE_NAMES,
    operator_names=tuple(cfg[0] for cfg in OPS_CONFIG),
)

ADVANCED_VOCAB = FormulaVocab(
    feature_names=ADVANCED_FEATURE_NAMES,
    operator_names=tuple(cfg[0] for cfg in OPS_CONFIG),
)
