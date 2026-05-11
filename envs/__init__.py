__all__ = ["OffloadingEnv", "StarGroundEnv", "normalize_actions"]


def __getattr__(name):
    if name in __all__:
        from envs.offloading_env import OffloadingEnv, StarGroundEnv, normalize_actions

        values = {
            "OffloadingEnv": OffloadingEnv,
            "StarGroundEnv": StarGroundEnv,
            "normalize_actions": normalize_actions,
        }
        return values[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
