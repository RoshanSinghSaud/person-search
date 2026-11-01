"""Model adapters: OSNet (photo queries) and IRRA (text queries).

Encoders are imported lazily by :mod:`person_search.factory` so that pure-logic
modules (scoring, fusion, eval) do not pull in torch/CLIP at import time.
"""

__all__ = ["IRRAEncoder", "OSNetEncoder"]


def __getattr__(name: str) -> type:  # PEP 562 lazy exports
    if name == "OSNetEncoder":
        from person_search.models.osnet_encoder import OSNetEncoder

        return OSNetEncoder
    if name == "IRRAEncoder":
        from person_search.models.irra_encoder import IRRAEncoder

        return IRRAEncoder
    raise AttributeError(name)
