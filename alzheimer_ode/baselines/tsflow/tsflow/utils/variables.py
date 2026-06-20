try:
    from enum import StrEnum
except ImportError:
    from enum import Enum
    class StrEnum(str, Enum):
        def __str__(self):
            return self.value

        def __format__(self, format_spec):
            return self.value.__format__(format_spec)

        @staticmethod
        def _generate_next_value_(name, start, count, last_values):
            return name.lower()

from enum import auto


class Setting(StrEnum):
    UNIVARIATE = auto()
    MULTIVARIATE = auto()


# class Setting(StrEnum):
#     UNIVARIATE = "univariate"
#     MULTIVARIATE = "multivariate"


class Prior(StrEnum):
    SN = auto()
    ISO = auto()
    OU = auto()
    SE = auto()
    PE = auto()


# class Prior(StrEnum):
#     SN = "sn"
#     ISO = "iso"
#     OU = "ou"
#     SE = "se"
#     PE = "pe"


season_lengths = {
    "H": 24,
    "D": 30,
    "1D": 30,
    "B": 30,
    "Y": 1,
}

season_lengths_gluonts = {
    "S": 3600,  # 1 hour
    "T": 1440,  # 1 day
    "H": 24,  # 1 day
    "1D": 1,
    "D": 1,  # 1 day
    "W": 1,  # 1 week
    "M": 12,
    "B": 5,
    "Q": 4,
    "Y": 1,
}


def get_season_length(freq):
    return season_lengths[freq]


def get_lags_for_freq(freq_str: str):
    if freq_str == "H":
        lags_seq = [24 * i for i in [1, 2, 3, 4, 5, 6, 7, 14, 21, 28]]
    elif freq_str == "B":
        # TODO: Fix lags for B
        lags_seq = [30 * i for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]
    elif freq_str == "1D":
        lags_seq = [30 * i for i in [1, 2, 3, 4, 5, 6, 7]]
    else:
        raise NotImplementedError(f"Lags for {freq_str} are not implemented yet.")
    return lags_seq
