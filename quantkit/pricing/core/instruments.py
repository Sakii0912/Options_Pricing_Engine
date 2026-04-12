"""Option instrument definitions"""

from enum import Enum
from dataclasses import dataclass


class OptionType(Enum):
    """Option type: Call or Put"""
    CALL = "call"
    PUT = "put"


class OptionStyle(Enum):
    """Option exercise style: European or American"""
    EUROPEAN = "european"
    AMERICAN = "american"


@dataclass
class Option:
    """
    Option contract specification.
    
    Attributes:
        strike: Strike price (K)
        maturity: Time to maturity in years (T)
        option_type: OptionType.CALL or OptionType.PUT
        style: OptionStyle.EUROPEAN or OptionStyle.AMERICAN
    """
    strike: float
    maturity: float
    option_type: OptionType
    style: OptionStyle
