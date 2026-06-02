#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

from .FLAME import FLAMEModel


def __getattr__(name):
    if name == "RenderMesh":
        from .renderer_utils import RenderMesh

        return RenderMesh
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["FLAMEModel", "RenderMesh"]
