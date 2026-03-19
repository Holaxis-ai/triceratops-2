from __future__ import annotations

import numpy as np
import pytest

from triceratops.domain.entities import ExternalLightCurve, LightCurve
from triceratops.domain.result import ScenarioResult, ValidationResult
from triceratops.domain.scenario_id import ScenarioID
from triceratops.plotting import fits as fits_module
from triceratops.plotting.fits import (
    _plottable_scenarios,
    _tess_plot_light_curve,
    plot_fits_joint,
)


def _scenario_result(scenario_id: ScenarioID, *, host_star_tic_id: int) -> ScenarioResult:
    n = 4
    zeros = np.zeros(n)
    ones = np.ones(n)
    return ScenarioResult(
        scenario_id=scenario_id,
        host_star_tic_id=host_star_tic_id,
        ln_evidence=-10.0,
        host_mass_msun=ones.copy(),
        host_radius_rsun=ones.copy(),
        host_u1=zeros.copy(),
        host_u2=zeros.copy(),
        period_days=ones.copy(),
        inclination_deg=np.full(n, 89.0),
        impact_parameter=zeros.copy(),
        eccentricity=zeros.copy(),
        arg_periastron_deg=zeros.copy(),
        planet_radius_rearth=ones.copy(),
        eb_mass_msun=zeros.copy(),
        eb_radius_rsun=zeros.copy(),
        flux_ratio_eb_tess=zeros.copy(),
        companion_mass_msun=zeros.copy(),
        companion_radius_rsun=zeros.copy(),
        flux_ratio_companion_tess=zeros.copy(),
    )


def _validation_result(
    scenario_result: ScenarioResult,
    *,
    flux_ratio_map: dict[int, float] | None = None,
) -> ValidationResult:
    return ValidationResult(
        target_id=123,
        false_positive_probability=0.1,
        nearby_false_positive_probability=0.01,
        scenario_results=[scenario_result],
        host_star_flux_ratio_tess_by_tic_id={} if flux_ratio_map is None else flux_ratio_map,
    )


def _light_curve() -> LightCurve:
    return LightCurve(
        time_days=np.array([-0.01, 0.0, 0.01]),
        flux=np.array([1.0, 0.99, 1.0]),
        flux_err=0.001,
    )


def test_tess_plot_light_curve_renorms_nearby_rows() -> None:
    light_curve = _light_curve()
    scenario_result = _scenario_result(ScenarioID.NTP, host_star_tic_id=42)
    validation_result = _validation_result(scenario_result, flux_ratio_map={42: 0.25})

    plotted = _tess_plot_light_curve(light_curve, scenario_result, validation_result)

    np.testing.assert_allclose(
        plotted.flux,
        (light_curve.flux - 0.75) / 0.25,
    )
    assert plotted.flux_err == light_curve.flux_err / 0.25


def test_tess_plot_light_curve_renorms_target_rows_to_host_flux_ratio() -> None:
    light_curve = _light_curve()
    scenario_result = _scenario_result(ScenarioID.TP, host_star_tic_id=123)
    validation_result = _validation_result(scenario_result, flux_ratio_map={123: 0.5})

    plotted = _tess_plot_light_curve(light_curve, scenario_result, validation_result)

    np.testing.assert_allclose(
        plotted.flux,
        (light_curve.flux - 0.5) / 0.5,
    )
    assert plotted.flux_err == light_curve.flux_err / 0.5


def test_tess_plot_light_curve_falls_back_without_host_flux_ratio() -> None:
    light_curve = _light_curve()
    scenario_result = _scenario_result(ScenarioID.NEB, host_star_tic_id=99)
    validation_result = _validation_result(scenario_result, flux_ratio_map={})

    plotted = _tess_plot_light_curve(light_curve, scenario_result, validation_result)

    assert plotted is light_curve


def test_plottable_scenarios_prefers_high_probability_rows_when_truncated() -> None:
    low = _scenario_result(ScenarioID.NTP, host_star_tic_id=1)
    low.relative_probability = 0.1
    high = _scenario_result(ScenarioID.NTP, host_star_tic_id=2)
    high.relative_probability = 0.9
    validation_result = ValidationResult(
        target_id=123,
        false_positive_probability=0.1,
        nearby_false_positive_probability=0.01,
        scenario_results=[low, high],
    )

    grouped = _plottable_scenarios(validation_result, max_per_column=1)

    assert grouped[0] == [high]


def test_plot_fits_joint_uses_host_renormed_tess_curve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    light_curve = _light_curve()
    external_lc = ExternalLightCurve(light_curve=_light_curve(), band="J")
    scenario_result = _scenario_result(ScenarioID.NTP, host_star_tic_id=42)
    scenario_result.relative_probability = 0.5
    validation_result = _validation_result(scenario_result, flux_ratio_map={42: 0.25})
    seen_fluxes: list[np.ndarray] = []

    def _fake_best_fit_model(model_time, sr, lc, *, external_lc_index=None):
        _ = (sr, external_lc_index)
        seen_fluxes.append(np.asarray(lc.flux))
        return np.ones_like(model_time)

    monkeypatch.setattr(fits_module, "_best_fit_model", _fake_best_fit_model)
    monkeypatch.setattr(fits_module, "_finalize_plot", lambda *args, **kwargs: None)

    plot_fits_joint(
        light_curve,
        external_lc,
        validation_result,
        nrows=1,
        save=False,
    )

    assert len(seen_fluxes) == 2
    np.testing.assert_allclose(seen_fluxes[0], external_lc.light_curve.flux)
    np.testing.assert_allclose(
        seen_fluxes[1],
        (light_curve.flux - 0.75) / 0.25,
    )
