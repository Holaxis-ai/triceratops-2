"""Stellar field plotting, including original-style TPF/cutout views."""
from __future__ import annotations

from math import ceil, floor

import numpy as np

from triceratops.domain.entities import StellarField


def plot_field(
    stellar_field: StellarField,
    search_radius_px: int | float,
    save: bool = False,
    fname: str | None = None,
    *,
    pixel_coords: np.ndarray | None = None,
    aperture_pixels: np.ndarray | None = None,
    image: np.ndarray | None = None,
    sector: int | None = None,
) -> None:
    """Plot star positions in the photometric field around the target.

    When ``pixel_coords`` and ``image`` are supplied, this renders an
    original-style TRICERATOPS-plus field plot with a pixel-grid view and
    a sector cutout view. Otherwise it falls back to the simplified
    sky-coordinate field plot.

    Args:
        stellar_field: StellarField containing the target and all neighbour
            stars.  stars[0] is always the target.
        search_radius_px: Search radius in pixels.  Converted to arcsec using
            the TESS pixel scale (20.25 arcsec/pixel) for the radius circle.
        save: If True, save the figure to a file instead of showing it.
        fname: Output filename (without extension).  Ignored when ``save``
            is False.  If ``save`` is True and ``fname`` is None, a default
            name ``TIC<id>_field.pdf`` is used.
        pixel_coords: Optional per-star pixel coordinates shaped ``(N, 2)``
            as ``(col, row)`` for original-style field plots.
        aperture_pixels: Optional aperture pixel coordinates shaped ``(M, 2)``
            as ``(col, row)``.
        image: Optional sector cutout image shaped ``(rows, cols)``.
        sector: Optional sector label for the title/default filename.
    """
    if pixel_coords is not None and image is not None:
        _plot_tpf_field(
            stellar_field,
            search_radius_px,
            pixel_coords=pixel_coords,
            aperture_pixels=aperture_pixels,
            image=image,
            sector=sector,
            save=save,
            fname=fname,
        )
        return

    _plot_angular_field(
        stellar_field,
        search_radius_px,
        save=save,
        fname=fname,
    )


def _plot_angular_field(
    stellar_field: StellarField,
    search_radius_px: int | float,
    *,
    save: bool = False,
    fname: str | None = None,
) -> None:
    """Simplified sky-coordinate field plot."""
    import matplotlib.pyplot as plt
    from matplotlib import cm

    TESS_ARCSEC_PER_PIX = 20.25
    search_radius_arcsec = search_radius_px * TESS_ARCSEC_PER_PIX

    target = stellar_field.target
    stars = stellar_field.stars

    # Build arrays for plotting.
    # Position angles are East of North; convert to (x_arcsec, y_arcsec):
    #   x = sep * sin(PA)  (East is positive x)
    #   y = sep * cos(PA)  (North is positive y)
    pa_rad = np.array([np.deg2rad(s.position_angle_deg) for s in stars])
    sep = np.array([s.separation_arcsec for s in stars])
    x = sep * np.sin(pa_rad)
    y = sep * np.cos(pa_rad)
    tmags = np.array([s.tmag for s in stars])

    vmin = floor(float(np.nanmin(tmags)))
    vmax = ceil(float(np.nanmax(tmags)))

    fig, ax = plt.subplots(figsize=(7, 6.5))
    plt.subplots_adjust(right=0.88)

    # Search radius circle (dashed)
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(
        search_radius_arcsec * np.sin(theta),
        search_radius_arcsec * np.cos(theta),
        "k--",
        alpha=0.5,
        zorder=0,
        label=f"Search radius ({search_radius_px} px)",
    )

    # Neighbour stars (circles, scaled by marker area)
    if len(stars) > 1:
        sc = ax.scatter(
            x[1:],
            y[1:],
            c=tmags[1:],
            s=75,
            edgecolors="k",
            cmap=cm.viridis_r,
            vmin=vmin,
            vmax=vmax,
            zorder=2,
            rasterized=True,
            label="Neighbour stars",
        )
    else:
        # No neighbours -- create a dummy scatter for the colourbar
        sc = ax.scatter(
            [], [],
            c=[],
            cmap=cm.viridis_r,
            vmin=vmin,
            vmax=vmax,
        )

    # Target star (larger star marker)
    ax.scatter(
        [x[0]],
        [y[0]],
        c=[tmags[0]],
        s=250,
        marker="*",
        edgecolors="k",
        cmap=cm.viridis_r,
        vmin=vmin,
        vmax=vmax,
        zorder=3,
        label=f"Target (TIC {target.tic_id})",
    )

    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.ax.set_ylabel("TESS mag", rotation=270, fontsize=12, labelpad=18)

    ax.set_xlabel("East offset (arcsec)", fontsize=12)
    ax.set_ylabel("North offset (arcsec)", fontsize=12)
    ax.set_title(f"Stellar field — TIC {target.tic_id}", fontsize=13)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_aspect("equal")

    # Compass arrows: N up, E right (already in our coord system)
    arrow_len = search_radius_arcsec * 0.15
    ax.annotate(
        "", xy=(arrow_len, 0), xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", color="k"),
    )
    ax.text(arrow_len * 1.15, 0, "E", ha="left", va="center", fontsize=10)
    ax.annotate(
        "", xy=(0, arrow_len), xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", color="k"),
    )
    ax.text(0, arrow_len * 1.15, "N", ha="center", va="bottom", fontsize=10)

    if save:
        plt.tight_layout()
        if fname is None:
            fname = f"TIC{target.tic_id}_field"
        plt.savefig(f"{fname}.pdf")
    else:
        plt.tight_layout()
        plt.show()
    plt.close(fig)


def _plot_tpf_field(
    stellar_field: StellarField,
    search_radius_px: int | float,
    *,
    pixel_coords: np.ndarray,
    aperture_pixels: np.ndarray | None,
    image: np.ndarray,
    sector: int | None,
    save: bool,
    fname: str | None,
) -> None:
    """Original-style field plot using pixel coordinates and a cutout image."""
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredDirectionArrows

    coords = np.asarray(pixel_coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(
            f"pixel_coords must have shape (N, 2), got {coords.shape}"
        )
    img = np.asarray(image, dtype=float)
    if img.ndim != 2:
        raise ValueError(f"image must have shape (rows, cols), got {img.shape}")
    aperture = (
        np.empty((0, 2), dtype=float)
        if aperture_pixels is None
        else np.asarray(aperture_pixels, dtype=float)
    )

    n_rows, n_cols = img.shape
    corners_x = np.arange(-0.5, n_cols + 0.5, 1.0)
    corners_y = np.arange(-0.5, n_rows + 0.5, 1.0)
    centers_x = np.arange(0, n_cols, 1.0)
    centers_y = np.arange(0, n_rows, 1.0)
    tmags = np.array([star.tmag for star in stellar_field.stars], dtype=float)
    vmin = floor(float(np.nanmin(tmags)))
    vmax = ceil(float(np.nanmax(tmags)))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    plt.subplots_adjust(right=0.9)

    _draw_aperture_outline(axes[0], aperture, color="red")
    for x in corners_x:
        axes[0].plot(np.full_like(corners_y, x), corners_y, "k-", lw=0.5, zorder=0)
    for y in corners_y:
        axes[0].plot(corners_x, np.full_like(corners_x, y), "k-", lw=0.5, zorder=0)

    theta = np.linspace(0, 2 * np.pi, 200)
    axes[0].plot(
        coords[0, 0] + search_radius_px * np.cos(theta),
        coords[0, 1] + search_radius_px * np.sin(theta),
        "k--",
        alpha=0.5,
        zorder=0,
    )

    if len(coords) > 1:
        sc = axes[0].scatter(
            coords[1:, 0],
            coords[1:, 1],
            c=tmags[1:],
            s=75,
            edgecolors="k",
            cmap=cm.viridis_r,
            vmin=vmin,
            vmax=vmax,
            zorder=2,
            rasterized=True,
        )
    else:
        sc = axes[0].scatter(
            [], [],
            c=[],
            cmap=cm.viridis_r,
            vmin=vmin,
            vmax=vmax,
        )
    axes[0].scatter(
        [coords[0, 0]],
        [coords[0, 1]],
        c=[tmags[0]],
        s=250,
        marker="*",
        edgecolors="k",
        cmap=cm.viridis_r,
        vmin=vmin,
        vmax=vmax,
        zorder=3,
    )
    cb1 = fig.colorbar(sc, ax=axes[0], pad=0.02)
    cb1.ax.set_ylabel("TESS mag", rotation=270, fontsize=12, labelpad=18)
    axes[0].set_xlim(float(np.min(corners_x)), float(np.max(corners_x)))
    axes[0].set_ylim(float(np.min(corners_y)), float(np.max(corners_y)))
    axes[0].set_xticks(centers_x)
    axes[0].set_yticks(centers_y)
    axes[0].tick_params(width=0)
    axes[0].tick_params(axis="x", labelrotation=90)
    axes[0].set_ylabel("pixel row number", fontsize=12)
    axes[0].set_xlabel("pixel column number", fontsize=12)
    if len(coords) > 1:
        v1 = np.array([0.0, 1.0])
        v2 = coords[1] - coords[0]
        if np.linalg.norm(v2) > 0:
            sign = np.sign(v2[0])
            angle1 = sign * (
                np.arccos(np.dot(v1, v2) / np.sqrt(np.dot(v1, v1) * np.dot(v2, v2)))
                * 180
                / np.pi
            )
            angle2 = stellar_field.stars[1].position_angle_deg
            rot = angle1 - angle2
            rotated_arrow = AnchoredDirectionArrows(
                axes[0].transAxes,
                "E",
                "N",
                loc="upper left",
                color="k",
                angle=-rot,
                length=0.1,
                fontsize=0.05,
                back_length=0,
                head_length=5,
                head_width=5,
                tail_width=1,
            )
            axes[0].add_artist(rotated_arrow)

    im = axes[1].imshow(
        img,
        extent=[-0.5, n_cols - 0.5, n_rows - 0.5, -0.5],
    )
    _draw_aperture_outline(axes[1], aperture, color="red")
    axes[1].set_xlim(-0.5, n_cols - 0.5)
    axes[1].set_ylim(-0.5, n_rows - 0.5)
    axes[1].set_xticks(centers_x)
    axes[1].set_yticks(centers_y)
    axes[1].tick_params(width=0)
    axes[1].tick_params(axis="x", labelrotation=90)
    axes[1].set_ylabel("pixel row number", fontsize=12)
    axes[1].set_xlabel("pixel column number", fontsize=12)
    cb2 = fig.colorbar(im, ax=axes[1], pad=0.02)
    cb2.ax.set_ylabel("flux [e$^-$ s$^{-1}$]", rotation=270, fontsize=12, labelpad=18)

    plt.tight_layout()
    if save:
        if fname is None:
            suffix = "" if sector is None else f"_sector{sector}"
            fname = f"TIC{stellar_field.target.tic_id}{suffix}"
        plt.savefig(f"{fname}.pdf")
    else:
        plt.show()
    plt.close(fig)


def _draw_aperture_outline(ax, aperture_pixels: np.ndarray, *, color: str) -> None:
    for col, row in aperture_pixels:
        ax.plot([col - 0.5, col + 0.5], [row - 0.5, row - 0.5], color=color, zorder=2)
        ax.plot([col - 0.5, col + 0.5], [row + 0.5, row + 0.5], color=color, zorder=2)
        ax.plot([col - 0.5, col - 0.5], [row - 0.5, row + 0.5], color=color, zorder=2)
        ax.plot([col + 0.5, col + 0.5], [row - 0.5, row + 0.5], color=color, zorder=2)
