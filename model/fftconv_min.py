try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover - allow import without torch
    torch = None
    nn = None


if nn is None:
    class FFTConv2d:  # type: ignore[no-redef]
        """
        Torch-free placeholder to allow module import in minimal environments.
        """

        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("FFTConv2d requires torch to be installed.")
else:
    class FFTConv2d(nn.Module):
        """
        Minimal FFTConv2d-compatible wrapper.
        This is a safe, dependency-free fallback using standard Conv2d.
        """

        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size,
            padding=0,
            stride=1,
            dilation=1,
            groups=1,
            bias=True,
        ) -> None:
            super().__init__()
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.conv(x)
