from __future__ import annotations

from typing import Any

import matplotlib.artist as martist
import matplotlib.colors as mcolors
import matplotlib.legend as mlegend
import matplotlib.patches as mpatches
import matplotlib.text as mtext
import matplotlib.transforms as mtransforms
from matplotlib.legend_handler import HandlerBase


def fmt5(x, *, prepend=True):
    """Format a number to a 5-character string with scientific notation if necessary."""
    # Count integer digits (ignore sign)
    int_digits = len(str(int(abs(x))))
    # Determine decimals: ≤2 int_digits→2; 3→1; ≥4→0
    decimals = max(0, min(2, 4 - int_digits))
    # Fixed-point with the chosen decimals
    s_fixed = f"{x:.{decimals}f}"
    # Remove trailing ".0" if no decimals
    if decimals == 0 and "." in s_fixed:
        s_fixed = s_fixed.split(".", 1)[0]
    # Check if fixed-point would round a tiny non-zero to zero
    tiny_nonzero = x != 0 and decimals > 0 and abs(x) < 10 ** (-decimals)
    # Use fixed-point if it fits and isn't tiny non-zero
    if not tiny_nonzero and len(s_fixed) <= 5:
        return s_fixed.rjust(5) if prepend else s_fixed.ljust(5)
    # Try integer-only (skip this for tiny non-zero)
    if not tiny_nonzero:
        s_int = str(int(x))
        if len(s_int) <= 5:
            return s_int.rjust(5) if prepend else s_int.ljust(5)
    # Scientific notation, trimmed exponent
    mant, exp = f"{x:.0e}".split("e")
    s_sci = f"{mant}e{int(exp)}"
    if len(s_sci) <= 5:
        return s_sci.rjust(5) if prepend else s_sci.ljust(5)
    # Last-ditch: single-digit mantissa
    s_sci2 = f"{mant[0]}e{int(exp)}"[-5:]
    return s_sci2.rjust(5) if prepend else s_sci2.ljust(5)


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
