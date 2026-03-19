"""Best-fit light-curve plots for validation scenarios."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from triceratops.domain.entities import ExternalLightCurve, LightCurve
from triceratops.domain.result import ScenarioResult, ValidationResult
from triceratops.domain.scenario_id import ScenarioID

_MIN_PROB = 1e-10

_TP_SCENARIOS: frozenset[ScenarioID] = frozenset({
    ScenarioID.TP, ScenarioID.PTP, ScenarioID.STP,
    ScenarioID.DTP, ScenarioID.BTP, ScenarioID.NTP,
})
_EB_SCENARIOS: frozenset[ScenarioID] = frozenset({
    ScenarioID.EB, ScenarioID.PEB, ScenarioID.SEB,
    ScenarioID.DEB, ScenarioID.BEB, ScenarioID.NEB,
})
_EBX2P_SCENARIOS: frozenset[ScenarioID] = frozenset({
    ScenarioID.EBX2P, ScenarioID.PEBX2P, ScenarioID.SEBX2P,
    ScenarioID.DEBX2P, ScenarioID.BEBX2P, ScenarioID.NEBX2P,
})


def _column_for_scenario(sid: ScenarioID) -> int:
    if sid in _TP_SCENARIOS:
        return 0
    if sid in _EB_SCENARIOS:
        return 1
    return 2


def _is_companion_scenario(sid: ScenarioID) -> bool:
    return sid in frozenset({
        ScenarioID.STP, ScenarioID.SEB, ScenarioID.SEBX2P,
        ScenarioID.BTP, ScenarioID.BEB, ScenarioID.BEBX2P,
    })


def _median_or(default: np.ndarray, values: Sequence[np.ndarray], index: int) -> float:
    if index < len(values) and len(values[index]) > 0:
        return float(np.median(values[index]))
    return float(np.median(default))


def _best_fit_model(
    model_time: np.ndarray,
    scenario_result: ScenarioResult,
    light_curve: LightCurve,
    *,
    external_lc_index: int | None = None,
) -> np.ndarray:
    """Compute a representative best-fit model for one scenario."""
    from triceratops.likelihoods.geometry import semi_major_axis
    from triceratops.likelihoods.transit_model import (
        simulate_eb_transit,
        simulate_planet_transit,
    )

    sr = scenario_result
    sid = sr.scenario_id

    if float(np.median(sr.host_mass_msun)) == 0.0:
        return np.ones(len(model_time))

    M_s = float(np.median(sr.host_mass_msun))
    R_s = float(np.median(sr.host_radius_rsun))
    P_orb = float(np.median(sr.period_days))
    inc = float(np.median(sr.inclination_deg))
    ecc = float(np.median(sr.eccentricity))
    argp = float(np.median(sr.arg_periastron_deg))

    if external_lc_index is None:
        u1 = float(np.median(sr.host_u1))
        u2 = float(np.median(sr.host_u2))
        fr_comp = float(np.median(sr.flux_ratio_companion_tess))
        fr_eb = float(np.median(sr.flux_ratio_eb_tess))
    else:
        u1 = _median_or(sr.host_u1, sr.external_lc_u1, external_lc_index)
        u2 = _median_or(sr.host_u2, sr.external_lc_u2, external_lc_index)
        fr_comp = _median_or(
            sr.flux_ratio_companion_tess,
            sr.external_lc_flux_ratio_comp,
            external_lc_index,
        )
        fr_eb = _median_or(
            sr.flux_ratio_eb_tess,
            sr.external_lc_flux_ratio_eb,
            external_lc_index,
        )

    companion_is_host = _is_companion_scenario(sid)
    is_eb = sid in ScenarioID.eb_scenarios()

    if is_eb:
        R_eb = float(np.median(sr.eb_radius_rsun))
        M_eb = float(np.median(sr.eb_mass_msun))
        a = float(semi_major_axis(np.array([P_orb]), M_s + M_eb)[0])
        flux, _ = simulate_eb_transit(
            time=model_time,
            rs=R_s,
            rcomp=R_eb,
            eb_flux_ratio=fr_eb,
            period=P_orb,
            inc=inc,
            a=a,
            u1=u1,
            u2=u2,
            ecc=ecc,
            argp=argp,
            companion_flux_ratio=fr_comp,
            companion_is_host=companion_is_host,
            exptime=light_curve.cadence_days,
            nsamples=light_curve.supersampling_rate,
        )
        return np.asarray(flux)

    R_p = float(np.median(sr.planet_radius_rearth))
    a = float(semi_major_axis(np.array([P_orb]), M_s)[0])
    flux = simulate_planet_transit(
        time=model_time,
        rp=R_p,
        period=P_orb,
        inc=inc,
        a=a,
        rs=R_s,
        u1=u1,
        u2=u2,
        ecc=ecc,
        argp=argp,
        companion_flux_ratio=fr_comp,
        companion_is_host=companion_is_host,
        exptime=light_curve.cadence_days,
        nsamples=light_curve.supersampling_rate,
    )
    return np.asarray(flux)


def _plottable_scenarios(
    validation_result: ValidationResult,
    *,
    max_per_column: int | None = None,
) -> list[list[ScenarioResult]]:
    has_any_visible = any(
        result.relative_probability >= _MIN_PROB
        for result in validation_result.scenario_results
    )
    if not has_any_visible:
        return [[], [], []]

    columns: list[list[ScenarioResult]] = [[], [], []]
    for result in validation_result.scenario_results:
        if len(result.host_mass_msun) == 0:
            continue
        column = _column_for_scenario(result.scenario_id)
        columns[column].append(result)
    for idx, column_results in enumerate(columns):
        column_results.sort(
            key=lambda result: result.relative_probability,
            reverse=True,
        )
        if max_per_column is not None:
            columns[idx] = column_results[:max_per_column]
    return columns


def _render_empty_plot(
    *,
    save: bool,
    fname: str | None,
    default_name: str,
    message: str,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=12,
    )
    ax.axis("off")
    if save:
        plt.savefig(f"{fname or default_name}.pdf")
    else:
        plt.tight_layout()
        plt.show()
    plt.close(fig)


def _build_axes(nrows: int):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(nrows, 3, figsize=(12, max(nrows, 1) * 4), sharex=False)
    if nrows == 1:
        axes = np.array([axes])
    return fig, axes


def _annotate_axis(ax, sr: ScenarioResult, validation_result: ValidationResult) -> None:
    if sr.host_star_tic_id != 0:
        host_label = str(sr.host_star_tic_id)
    elif sr.scenario_id in ScenarioID.nearby_scenarios():
        host_label = "unknown"
    else:
        host_label = str(validation_result.target_id)
    ax.annotate(host_label, xy=(0.05, 0.92), xycoords="axes fraction", fontsize=12)
    ax.annotate(str(sr.scenario_id), xy=(0.05, 0.05), xycoords="axes fraction", fontsize=12)


def _tess_plot_light_curve(
    light_curve: LightCurve,
    scenario_result: ScenarioResult,
    validation_result: ValidationResult,
) -> LightCurve:
    """Return the host-renormalized TESS light curve for one plotted row.

    Vendor TRICERATOPS+ renormalizes the plotted TESS light curve to the row
    host before overlaying the best-fit model. We mirror that behavior using
    the host TIC ID carried on the ScenarioResult and the flux-ratio map
    captured on ValidationResult during aggregation.
    """
    flux_ratio = validation_result.host_star_flux_ratio_tess_by_tic_id.get(
        scenario_result.host_star_tic_id,
    )
    if flux_ratio is None or flux_ratio <= 0.0 or flux_ratio > 1.0:
        return light_curve
    return light_curve.with_renorm(flux_ratio)


def _finalize_plot(fig, *, save: bool, fname: str | None, default_name: str) -> None:
    import matplotlib.pyplot as plt

    if save:
        plt.tight_layout()
        plt.savefig(f"{fname or default_name}.pdf")
    else:
        plt.tight_layout()
        plt.show()
    plt.close(fig)


def plot_fits(
    light_curve: LightCurve,
    validation_result: ValidationResult,
    save: bool = False,
    fname: str | None = None,
) -> None:
    """Plot TESS best-fit models for all non-negligible scenarios."""
    from matplotlib import ticker

    grouped = _plottable_scenarios(validation_result)
    nrows = max((len(column) for column in grouped), default=0)
    if nrows == 0:
        _render_empty_plot(
            save=save,
            fname=fname,
            default_name=f"TIC{validation_result.target_id}_fits",
            message="No scenarios with relative_probability ≥ 1e-10",
        )
        return

    model_time = np.linspace(
        float(np.min(light_curve.time_days)),
        float(np.max(light_curve.time_days)),
        200,
    )
    fig, axes = _build_axes(nrows)

    for column_index, column_results in enumerate(grouped):
        for row_index in range(nrows):
            ax = axes[row_index, column_index]
            if row_index >= len(column_results):
                ax.axis("off")
                continue
            sr = column_results[row_index]
            plotted_light_curve = _tess_plot_light_curve(
                light_curve,
                sr,
                validation_result,
            )
            y_formatter = ticker.ScalarFormatter(useOffset=False)
            ax.yaxis.set_major_formatter(y_formatter)
            ax.errorbar(
                plotted_light_curve.time_days,
                plotted_light_curve.flux,
                plotted_light_curve.flux_err,
                fmt="o",
                color="dodgerblue",
                elinewidth=1.0,
                capsize=0,
                markeredgecolor="black",
                alpha=0.25,
                zorder=0,
                rasterized=True,
            )
            try:
                model_flux = _best_fit_model(model_time, sr, plotted_light_curve)
            except Exception:  # noqa: BLE001
                model_flux = np.ones(len(model_time))
            ax.plot(model_time, model_flux, "k-", lw=3, zorder=2)
            ax.set_ylabel("normalized flux", fontsize=12)
            if row_index == nrows - 1:
                ax.set_xlabel("days from transit center", fontsize=12)
            _annotate_axis(ax, sr, validation_result)

    _finalize_plot(
        fig,
        save=save,
        fname=fname,
        default_name=f"TIC{validation_result.target_id}_fits",
    )


def plot_fits_palomar(
    external_light_curve: ExternalLightCurve,
    validation_result: ValidationResult,
    *,
    reference_light_curve: LightCurve | None = None,
    external_lc_index: int = 0,
    x_range: Sequence[float] | None = None,
    y_range: Sequence[float] | None = None,
    nrows: int = 0,
    save: bool = False,
    fname: str | None = None,
) -> None:
    """Plot best-fit models against one external follow-up light curve."""
    from matplotlib import ticker

    max_per_column = None if nrows <= 0 else nrows
    grouped = _plottable_scenarios(validation_result, max_per_column=max_per_column)
    panel_rows = max((len(column) for column in grouped), default=0)
    if panel_rows == 0:
        _render_empty_plot(
            save=save,
            fname=fname,
            default_name=f"TIC{validation_result.target_id}_fits_palomar",
            message="No scenarios with relative_probability ≥ 1e-10",
        )
        return

    lc = external_light_curve.light_curve
    if x_range is None or len(x_range) == 0:
        reference_time = (
            lc.time_days
            if reference_light_curve is None
            else reference_light_curve.time_days
        )
        model_time = np.linspace(
            float(np.min(reference_time)),
            float(np.max(reference_time)),
            200,
        )
    else:
        model_time = np.linspace(float(x_range[0]), float(x_range[1]), 200)

    fig, axes = _build_axes(panel_rows)
    flux_err = lc.flux_err if np.ndim(lc.flux_err) > 0 else float(lc.flux_err)

    for column_index, column_results in enumerate(grouped):
        for row_index in range(panel_rows):
            ax = axes[row_index, column_index]
            if row_index >= len(column_results):
                ax.axis("off")
                continue
            sr = column_results[row_index]
            y_formatter = ticker.ScalarFormatter(useOffset=False)
            ax.yaxis.set_major_formatter(y_formatter)
            ax.errorbar(
                lc.time_days,
                lc.flux,
                flux_err,
                fmt="o",
                color="red",
                elinewidth=1.0,
                capsize=0,
                markeredgecolor="black",
                alpha=0.25,
                zorder=0,
                rasterized=True,
            )
            try:
                model_flux = _best_fit_model(
                    model_time,
                    sr,
                    lc,
                    external_lc_index=external_lc_index,
                )
            except Exception:  # noqa: BLE001
                model_flux = np.ones(len(model_time))
            ax.plot(model_time, model_flux, "k-", lw=3, zorder=2)
            ax.set_ylabel("normalized flux", fontsize=12)
            if x_range is not None and len(x_range) == 2:
                ax.set_xlim(float(x_range[0]), float(x_range[1]))
            if y_range is not None and len(y_range) == 2:
                ax.set_ylim(float(y_range[0]), float(y_range[1]))
            if row_index == panel_rows - 1:
                ax.set_xlabel("days from transit center", fontsize=12)
            _annotate_axis(ax, sr, validation_result)

    _finalize_plot(
        fig,
        save=save,
        fname=fname,
        default_name=f"TIC{validation_result.target_id}_fits_palomar",
    )


def plot_fits_joint(
    light_curve: LightCurve,
    external_light_curve: ExternalLightCurve,
    validation_result: ValidationResult,
    *,
    external_lc_index: int = 0,
    x_range: Sequence[float] | None = None,
    y_range: Sequence[float] | None = None,
    nrows: int = 0,
    save: bool = False,
    fname: str | None = None,
) -> None:
    """Plot external-band and TESS best-fit models together."""
    from matplotlib import ticker

    max_per_column = None if nrows <= 0 else nrows
    grouped = _plottable_scenarios(validation_result, max_per_column=max_per_column)
    panel_rows = max((len(column) for column in grouped), default=0)
    if panel_rows == 0:
        _render_empty_plot(
            save=save,
            fname=fname,
            default_name=f"TIC{validation_result.target_id}_fits_joint",
            message="No scenarios with relative_probability ≥ 1e-10",
        )
        return

    lc = external_light_curve.light_curve
    if x_range is None or len(x_range) == 0:
        model_time = np.linspace(
            float(np.min(light_curve.time_days)),
            float(np.max(light_curve.time_days)),
            200,
        )
    else:
        model_time = np.linspace(float(x_range[0]), float(x_range[1]), 200)

    fig, axes = _build_axes(panel_rows)
    flux_err = lc.flux_err if np.ndim(lc.flux_err) > 0 else float(lc.flux_err)

    for column_index, column_results in enumerate(grouped):
        for row_index in range(panel_rows):
            ax = axes[row_index, column_index]
            if row_index >= len(column_results):
                ax.axis("off")
                continue
            sr = column_results[row_index]
            plotted_tess_light_curve = _tess_plot_light_curve(
                light_curve,
                sr,
                validation_result,
            )
            y_formatter = ticker.ScalarFormatter(useOffset=False)
            ax.yaxis.set_major_formatter(y_formatter)
            ax.errorbar(
                lc.time_days,
                lc.flux,
                flux_err,
                fmt="o",
                color="red",
                elinewidth=1.0,
                capsize=0,
                markeredgecolor="black",
                alpha=0.25,
                zorder=0,
                rasterized=True,
            )
            try:
                external_model = _best_fit_model(
                    model_time,
                    sr,
                    lc,
                    external_lc_index=external_lc_index,
                )
                tess_model = _best_fit_model(
                    model_time,
                    sr,
                    plotted_tess_light_curve,
                )
            except Exception:  # noqa: BLE001
                external_model = np.ones(len(model_time))
                tess_model = np.ones(len(model_time))
            ax.plot(model_time, external_model, "k-", lw=3, zorder=2)
            ax.plot(model_time, tess_model, "b-", lw=3, alpha=0.5, zorder=2)
            ax.set_ylabel("normalized flux", fontsize=12)
            if x_range is not None and len(x_range) == 2:
                ax.set_xlim(float(x_range[0]), float(x_range[1]))
            if y_range is not None and len(y_range) == 2:
                ax.set_ylim(float(y_range[0]), float(y_range[1]))
            if row_index == panel_rows - 1:
                ax.set_xlabel("days from transit center", fontsize=12)
            _annotate_axis(ax, sr, validation_result)

    _finalize_plot(
        fig,
        save=save,
        fname=fname,
        default_name=f"TIC{validation_result.target_id}_fits_joint",
    )
