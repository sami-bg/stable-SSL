from dataclasses import dataclass
from torchvision.transforms import v2
import torch


@dataclass
class TransformsConfig:
    """
    Configuration for the data used for training the model.

    Parameters:
    -----------
    transforms : list[tuple[str, dict]]
        The transformations to apply. For example:
        ```
        "RandomResizedCrop":{}
        ```
        for the default one, or
        ```
        "RandomResizedCrop":{"ratio":(0.3,2)}
        ```
        for custom settings.
    """

    name: str = None
    transforms: list[dict] = None

    def __post_init__(self):
        extra = [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
        if self.transforms is None:
            self.transforms = [{}]
            self._transform = v2.Compose(extra)
        else:
            self._transform = v2.Compose(
                [TransformConfig(**t) for t in self.transforms] + extra
            )

    def __call__(self, x):
        return self._transform(x)


@dataclass
class TransformConfig:
    """
    Configuration for the data used for training the model.

    Parameters:
    -----------
    transforms : list[tuple[str, dict]]
        The transformations to apply. For example:
        ```
        "RandomResizedCrop":{}
        ```
        for the default one, or
        ```
        "RandomResizedCrop":{"ratio":(0.3,2)}
        ```
        for custom settings.
    """

    name: str
    args: list
    kwargs: dict
    p: float = 1.0

    def __init__(self, name, args=None, kwargs=None, p=1):
        self.name = name
        self.args = args or []
        self.kwargs = kwargs or {}
        self.p = p
        if self.name is not None:
            t = v2.__dict__[self.name](*self.args, **self.kwargs)
            if self.p < 1:
                self._transform = v2.RandomApply(torch.nn.ModuleList([t]), p=self.p)
            else:
                self._transform = t
        else:
            self._transform = v2.Identity()

    def __call__(self, x):
        return self._transform(x)
