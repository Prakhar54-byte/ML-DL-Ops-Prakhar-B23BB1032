def read_video(*args, **kwargs):
    raise ImportError(
        "torchvision video decoding is unavailable in this environment. "
        "Install a compatible torchvision build if you need video support."
    )
