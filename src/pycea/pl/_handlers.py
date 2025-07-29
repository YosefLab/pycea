from __future__ import annotations

from typing import Any

import matplotlib.artist as martist
import matplotlib.colors as mcolors
import matplotlib.legend as mlegend
import matplotlib.patches as mpatches
import matplotlib.text as mtext
import matplotlib.transforms as mtransforms
from matplotlib.legend_handler import HandlerBase


def fmt5(x: float | None, *, prepend: bool = True):
    """
    Format *x* as an exactly-5-character string.

    • Tries fixed-point with two decimals.
    • Falls back to scientific notation, trimming the exponent so '1e5', '-2e9', …
    • Pads on the left (prepend=True) or right (prepend=False) with spaces.

    Parameters
    ----------
    x
        The number to format.
    prepend
        If `True`, pads on the left with spaces; if `False`, pads on the
    """
    # fixed-point, two decimals
    s = f"{x:.2f}"
    if len(s) <= 5:
        return s.rjust(5) if prepend else s.ljust(5)
    # scientific notation, exponent as integer → no leading zeros or '+'
    mant, exp = f"{x:.0e}".split("e")  # e.g. '1', '+05'
    s = f"{mant}e{int(exp)}"  # '1e5'
    if len(s) <= 5:
        return s.rjust(5) if prepend else s.ljust(5)
    # last resort: keep only the first mantissa digit
    mant = mant[0]  # '9' from '9.7'
    s = f"{mant}e{int(exp)}"  # '9e11', '-9e11', etc.
    # If still longer than 5, clip from the left; keep the last 5 chars
    s = s[-5:]
    return s.rjust(5) if prepend else s.ljust(5)


class HandlerColorbar(HandlerBase):
    """Legend handler that paints a tiny colorbar."""

    def __init__(
        self,
        cmap: mcolors.Colormap | Any,
        norm: mcolors.Normalize,
        *,
        N: int = 128,
        fmt: str = "{:.2g}",
        pad: int = 2,
        textprops: dict | None = None,
    ):
        """Create a colorbar legend handler.

        Parameters
        ----------
        cmap
            The colormap to use.
        norm
            The normalization to use.
        N
            The number of discrete colors to use.
        fmt
            The format string for the colorbar labels.
        pad
            The padding between the colorbar and the labels.
        textprops
            Additional properties to pass to the text labels.
        """
        super().__init__()
        self.cmap, self.norm = cmap, norm
        self.N = max(2, int(N))
        self.fmt = fmt
        self.pad = pad
        self.textprops = textprops or {}

    def create_artists(
        self,
        legend: mlegend.Legend,
        orig_handle: martist.Artist,
        xdescent: int,
        ydescent: int,
        width: int,
        height: int,
        fontsize: int,
        transform: mtransforms.Transform,
    ):
        """
        Return the legend artists generated.

        Parameters
        ----------
        legend : `~matplotlib.legend.Legend`
            The legend for which these legend artists are being created.
        orig_handle : `~matplotlib.artist.Artist` or similar
            The object for which these legend artists are being created.
        xdescent, ydescent, width, height : int
            The rectangle (*xdescent*, *ydescent*, *width*, *height*) that the
            legend artists being created should fit within.
        fontsize : int
            The fontsize in pixels. The legend artists being created should
            be scaled according to the given fontsize.
        trans : `~matplotlib.transforms.Transform`
            The transform that is applied to the legend artists being created.
            Typically from unit coordinates in the handler box to screen
            coordinates.
        """
        artists = []
        dw = (width - 50) / (self.N)
        for i in range(1, self.N):
            frac = i / (self.N - 1)
            colour = self.cmap(frac)
            artists.append(
                mpatches.Rectangle(
                    (xdescent + i * dw + 24, ydescent),
                    dw,
                    height,
                    transform=transform,
                    facecolor=colour,
                    edgecolor=colour,
                    lw=0,
                )
            )
        txt_kw = dict(self.textprops)
        txt_kw.setdefault("fontsize", fontsize * 0.8)
        txt_kw.setdefault("va", "center")
        vmin_txt = mtext.Text(
            xdescent - self.pad + 24,  # left of bar
            ydescent + 0.5 * height,
            fmt5(self.norm.vmin),
            ha="right",
            transform=transform,
            **txt_kw,
        )
        vmax_txt = mtext.Text(
            xdescent + width + self.pad - 25,  # right of bar
            ydescent + 0.5 * height,
            fmt5(self.norm.vmax, prepend=False),
            ha="left",
            transform=transform,
            **txt_kw,
        )
        artists.extend([vmin_txt, vmax_txt])
        return artists
