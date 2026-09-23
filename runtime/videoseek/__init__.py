__all__ = ["VideoSeekAgent"]


def __getattr__(name):
    if name == "VideoSeekAgent":
        from .agent import VideoSeekAgent

        return VideoSeekAgent
    raise AttributeError(name)
