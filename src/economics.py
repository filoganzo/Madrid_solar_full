"""
economics.py — what a panel is worth, and what it costs.

Two ideas drive everything in this file.

1. COST IS NOT LINEAR IN PANELS.
   A EUR/Wp headline is an average, and averages hide the decision. The
   inverter, the scaffold, the electrical board, the bidirectional meter and
   the legalisation paperwork cost the same whether you fit four panels or
   fourteen. The module, the rail, the DC string and the labour scale with
   panel count. Split that way, the first panel on a Madrid block roof costs
   roughly ten times what the twelfth costs. Which means:
     - small systems are structurally uneconomic in an apartment city,
     - the roof, not the panel, is the unit that has to clear the hurdle,
     - and the single biggest cost lever is not the module price, it is
       amortising the fixed cost over more panels, i.e. selling the whole
       roof to the whole building rather than one flat at a time.

2. VALUE IS NOT LINEAR IN KILOWATT-HOURS.
   Under RD 244/2019 a self-consumed kWh avoids the full retail price,
   including tolls and taxes, so it is worth 0.09-0.21 EUR depending on the
   hour. An exported kWh earns a compensation credit at about 0.05 EUR,
   capped monthly at the value of what you imported, and any surplus beyond
   that cap is given to the grid for nothing. So the marginal panel is worth
   progressively less: each one pushes more of the day's output past the
   household's own demand and into the cheap bucket.

   The consequence is the finding I would lead with in an interview: in Spain
   the economic value of a rooftop is set by the self-consumption fraction,
   not by the irradiance. Madrid has 1.35x London's sunshine and roughly the
   same payback, because the extra sun arrives at noon in July when nobody
   is home and the export price is near zero.

That is also the argument for Fuse's own micro-solar-plus-battery product
being the right shape for this market, arrived at from the arithmetic rather
than from the press release.
"""

from __future__ import annotations

import numpy as np

from .demand import month_index_from_doy, tariff_price


# --------------------------------------------------------------------------
# Capex
# --------------------------------------------------------------------------

def capex_breakdown(n_panels: int, cfg: dict, with_battery: bool = False,
                    apply_incentives: bool | None = None) -> dict:
    """Bill of materials for a system of n_panels, split fixed vs marginal."""
    c = cfg["capex"]
    inc = cfg["incentives"]
    if apply_incentives is None:
        apply_incentives = inc["include_in_base_case"]

    per_panel = (c["module_eur_per_panel"] + c["mounting_eur_per_panel"]
                 + c["dc_string_eur_per_panel"] + c["labour_eur_per_panel"])
    marginal = per_panel * n_panels

    licence = c["licence_eur"]
    if apply_incentives:
        licence *= (1.0 - inc["icio_bonus"])

    fixed = (c["inverter_eur"] + c["protections_meter_eur"]
             + c["roof_access_eur"] + c["engineering_cie_eur"] + licence)

    battery = cfg["battery"]["cost_eur"] if with_battery else 0.0

    subtotal = marginal + fixed + battery
    with_margin = subtotal * (1.0 + c["installer_margin"])
    total = with_margin * (1.0 + c["vat"])

    kwp = n_panels * cfg["panel"]["wp"] / 1000.0
    return {
        "n_panels": n_panels,
        "kwp": kwp,
        "marginal_eur": marginal,
        "fixed_eur": fixed,
        "battery_eur": battery,
        "subtotal_eur": subtotal,
        "total_eur": total,
        "eur_per_wp": total / (kwp * 1000.0) if kwp else float("nan"),
        "per_panel_marginal_eur": per_panel * (1 + c["installer_margin"])
                                  * (1 + c["vat"]),
        "fixed_share": (fixed * (1 + c["installer_margin"]) * (1 + c["vat"])
                        / total) if total else float("nan"),
    }


def marginal_cost_of_nth_panel(n: int, cfg: dict) -> float:
    """Cost of adding the nth panel to a system that already has n-1."""
    a = capex_breakdown(n, cfg)["total_eur"]
    b = capex_breakdown(n - 1, cfg)["total_eur"] if n > 1 else \
        capex_breakdown(0, cfg)["total_eur"]
    return a - b


# --------------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------------

def dispatch(gen_kwh: np.ndarray, load_kwh: np.ndarray, cfg: dict,
             with_battery: bool = False) -> dict:
    """Hour-by-hour allocation of generation to load, battery and grid.

    Greedy, not optimised: solar goes to load first, then to the battery,
    then to export; the battery discharges whenever load exceeds generation.
    A price-aware optimiser would earn a few percent more by holding charge
    for the 18:00-22:00 peak band, and that gap is itself interesting — it is
    exactly the value a trading and forecasting operation adds to a fleet of
    domestic batteries, and it is quantified in run_madrid.py rather than
    claimed.
    """
    gen = np.asarray(gen_kwh, dtype=float)
    load = np.asarray(load_kwh, dtype=float)

    direct = np.minimum(gen, load)
    surplus = gen - direct
    deficit = load - direct

    if not with_battery:
        return {
            "direct_kwh": direct,
            "battery_discharge_kwh": np.zeros_like(gen),
            "export_kwh": surplus,
            "import_kwh": deficit,
            "self_consumed_kwh": direct,
            "cycles": 0.0,
        }

    b = cfg["battery"]
    cap = b["usable_kwh"]
    eta_1way = np.sqrt(b["round_trip_efficiency"])
    soc = 0.0
    charge = np.zeros_like(gen)
    discharge = np.zeros_like(gen)
    throughput = 0.0

    for i in range(len(gen)):
        if surplus[i] > 0 and soc < cap:
            room = (cap - soc) / eta_1way
            c = min(surplus[i], room, b["max_charge_kw"])
            charge[i] = c
            soc += c * eta_1way
            surplus[i] -= c
        elif deficit[i] > 0 and soc > 0:
            avail = soc * eta_1way
            d = min(deficit[i], avail, b["max_discharge_kw"])
            discharge[i] = d
            soc -= d / eta_1way
            deficit[i] -= d
            throughput += d

    return {
        "direct_kwh": direct,
        "battery_charge_kwh": charge,
        "battery_discharge_kwh": discharge,
        "export_kwh": surplus,
        "import_kwh": deficit,
        "self_consumed_kwh": direct + discharge,
        "cycles": throughput / cap if cap else 0.0,
    }


# --------------------------------------------------------------------------
# Revenue
# --------------------------------------------------------------------------

def annual_value(disp: dict, cfg: dict, doy: np.ndarray,
                 hour: np.ndarray) -> dict:
    """Euros of bill reduction in year one.

    Applies the RD 244/2019 monthly cap properly: the export credit in a
    month cannot exceed the value of energy imported in that month. Getting
    this wrong is the standard error in Spanish solar sales material, and it
    flatters oversized systems — which are exactly the ones an installer
    wants to sell.
    """
    price = tariff_price(cfg, doy, hour)
    month = month_index_from_doy(doy)

    avoided = disp["self_consumed_kwh"] * price
    import_cost = disp["import_kwh"] * price
    export_gross = disp["export_kwh"] * cfg["export"]["compensation_eur_kwh"]

    credit = np.zeros(12)
    lost = np.zeros(12)
    for m in range(12):
        sel = month == m
        g = export_gross[sel].sum()
        cap = import_cost[sel].sum() if cfg["export"]["monthly_cap_is_energy_term"] \
            else np.inf
        credit[m] = min(g, cap)
        lost[m] = max(g - cap, 0.0)

    return {
        "avoided_import_eur": float(avoided.sum()),
        "export_credit_eur": float(credit.sum()),
        "export_credit_lost_to_cap_eur": float(lost.sum()),
        "total_benefit_eur": float(avoided.sum() + credit.sum()),
        "self_consumption_fraction": float(
            disp["self_consumed_kwh"].sum()
            / max(disp["self_consumed_kwh"].sum() + disp["export_kwh"].sum(),
                  1e-9)),
        "self_sufficiency_fraction": float(
            disp["self_consumed_kwh"].sum()
            / max(disp["self_consumed_kwh"].sum() + disp["import_kwh"].sum(),
                  1e-9)),
        "effective_value_per_kwh": float(
            (avoided.sum() + credit.sum())
            / max(disp["self_consumed_kwh"].sum() + disp["export_kwh"].sum(),
                  1e-9)),
    }


# --------------------------------------------------------------------------
# Returns
# --------------------------------------------------------------------------

def cashflows(capex_eur: float, year1_benefit_eur: float, cfg: dict,
              degradation: np.ndarray,
              battery_replacement_year: int | None = None,
              annual_opex_eur: float | None = None,
              inverter_replacement: bool = True) -> np.ndarray:
    """Real cash flows, year 0 to end of life.

    Opex is a parameter rather than a constant because the same 45 EUR/yr and
    the same mid-life inverter replacement cannot apply to both a 166 kWp
    roof installation and a plug-in balcony kit. In the first version they
    did, which made a 620 EUR kit look like a 26-year payback: the fixed
    maintenance charge was larger than the thing it was maintaining.
    """
    n = cfg["panel"]["life_years"]
    drift = (1.0 + cfg["finance"]["inflation_energy"]) ** np.arange(n)
    flows = year1_benefit_eur * degradation[:n] * drift
    if annual_opex_eur is None:
        annual_opex_eur = 45.0
    opex = np.full(n, float(annual_opex_eur))
    if inverter_replacement:
        opex[12] += cfg["capex"]["inverter_eur"] * 0.8
    if battery_replacement_year and battery_replacement_year < n:
        opex[battery_replacement_year] += cfg["battery"]["cost_eur"] * 0.6
    return np.concatenate([[-capex_eur], flows - opex])


def npv(flows: np.ndarray, rate: float) -> float:
    t = np.arange(len(flows))
    return float((flows / (1.0 + rate) ** t).sum())


def irr(flows: np.ndarray) -> float:
    """Bisection. Returns nan when no sign change exists, which is the honest
    answer for a project that never pays back rather than a made-up number."""
    if flows[0] >= 0 or flows[1:].sum() <= -flows[0]:
        if flows[1:].sum() <= -flows[0]:
            return float("nan")
    lo, hi = -0.9, 1.5
    f_lo, f_hi = npv(flows, lo), npv(flows, hi)
    if f_lo * f_hi > 0:
        return float("nan")
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if npv(flows, mid) * f_lo > 0:
            lo = mid
        else:
            hi = mid
    return float((lo + hi) / 2.0)


def simple_payback_years(flows: np.ndarray) -> float:
    cum = np.cumsum(flows)
    idx = np.where(cum >= 0)[0]
    if len(idx) == 0:
        return float("inf")
    i = idx[0]
    if i == 0:
        return 0.0
    prev = cum[i - 1]
    return float((i - 1) + (-prev) / flows[i])


def lcoe(capex_eur: float, annual_kwh_year1: float, cfg: dict,
         degradation: np.ndarray, rate: float | None = None) -> float:
    """Levelised cost of the energy this system produces, EUR/kWh.

    Worth computing because it is the number that can be compared directly
    with the retail tariff and with Fuse's own wholesale cost. If rooftop
    LCOE sits below the retail price but above the wholesale price plus
    network charges, the asset makes sense behind the meter and not in front
    of it. That single comparison decides who should own it.
    """
    if rate is None:
        rate = cfg["finance"]["household_discount_rate"]
    n = cfg["panel"]["life_years"]
    t = np.arange(1, n + 1)
    energy = (annual_kwh_year1 * degradation[:n] / (1.0 + rate) ** t).sum()
    opex = np.full(n, 45.0)
    opex[12] += cfg["capex"]["inverter_eur"] * 0.8
    cost = capex_eur + (opex / (1.0 + rate) ** t).sum()
    return float(cost / energy) if energy > 0 else float("nan")
