# -----------------------------------------------------------------------------
# Predbat Home Battery System
# Copyright Trefor Southwell 2026 - All Rights Reserved
# This application maybe used for personal use only and not for commercial use
# -----------------------------------------------------------------------------
# fmt off
# pylint: disable=consider-using-f-string
# pylint: disable=line-too-long
# pylint: disable=attribute-defined-outside-init

from types import SimpleNamespace

from const import PREDICT_STEP
from prediction import Prediction
from utils import derive_fleet_power_curves


def _inverter(soc, cap, reserve_percent=4, rate_kw=2.6):
    """
    Build an inverter dictionary with the fields the curve derivation reads
    """
    return {"soc_kw": soc, "soc_max": cap, "reserve": cap * reserve_percent / 100.0, "battery_rate_max_charge": rate_kw / 60.0, "battery_rate_max_discharge": rate_kw / 60.0}


def test_fleet_power_curves(my_predbat):
    """
    Test the derived multi-inverter power curves: single inverter bypass, depleted inverters dropping out, full inverters
    dropping out, held inverters, the per-battery taper and the sparse output
    """
    failed = False

    # One inverter, nothing to derive
    charge, discharge = derive_fleet_power_curves([_inverter(5.0, 10.0)], 5.0)
    if charge is not None or discharge is not None:
        print("ERROR: A single inverter should give no derived curves, got {} {}".format(charge, discharge))
        failed = True

    # Two identical inverters, half full: both deliver until both are at reserve, so the discharge curve is flat above reserve
    inverters = [_inverter(5.0, 10.0), _inverter(5.0, 10.0)]
    charge, discharge = derive_fleet_power_curves(inverters, 10.0)
    if discharge.get(50, 1.0) != 1.0 or discharge.get(10, 1.0) != 1.0:
        print("ERROR: Identical inverters should deliver the full rate above reserve, got {}".format(discharge))
        failed = True
    if discharge.get(4, 1.0) != 0.0 or discharge.get(0, 1.0) != 0.0:
        print("ERROR: Nothing should be delivered at reserve, got {}".format(discharge))
        failed = True

    # Unequal sizes, equal rates: the smaller battery runs out first and the fleet rate halves
    inverters = [_inverter(10.0, 10.0), _inverter(2.0, 2.0)]
    charge, discharge = derive_fleet_power_curves(inverters, 12.0)
    low = discharge.get(10, 1.0)
    if abs(low - 0.5) > 0.01:
        print("ERROR: With the small battery near empty only half the fleet rate should remain at 10%, got {}".format(low))
        failed = True

    # A full inverter stops charging: charge curve halves near the top for a pair where one is already full
    inverters = [_inverter(10.0, 10.0), _inverter(2.0, 10.0)]
    charge, discharge = derive_fleet_power_curves(inverters, 12.0)
    if abs(charge.get(70, 1.0) - 0.5) > 0.01:
        print("ERROR: With one inverter full only half the fleet charge rate should remain, got {}".format(charge.get(70, 1.0)))
        failed = True

    # An inverter held at zero discharge strands its energy: nothing is deliverable once the others reach reserve
    held = _inverter(8.0, 10.0)
    held["battery_rate_max_discharge"] = 0.0
    inverters = [_inverter(8.0, 10.0), held]
    charge, discharge = derive_fleet_power_curves(inverters, 16.0)
    if discharge.get(40, 1.0) != 0.0 or discharge.get(80, 1.0) != 1.0:
        print("ERROR: A held inverter should strand its energy, got {} at 40 and {} at 80".format(discharge.get(40, 1.0), discharge.get(80, 1.0)))
        failed = True

    # The per-battery taper is applied at each inverter's own SoC: charge taper at 100% halves the rate of a full inverter, which is already full so it drops out
    taper = {95: 0.5, 96: 0.5, 97: 0.5, 98: 0.5, 99: 0.5}
    inverters = [_inverter(9.7, 10.0), _inverter(5.0, 10.0)]
    charge, discharge = derive_fleet_power_curves(inverters, 14.7, battery_charge_curve=taper)
    if charge.get(97, 1.0) >= 1.0:
        print("ERROR: The taper should reduce the charge rate while an inverter is in it, got {}".format(charge.get(97, 1.0)))
        failed = True

    # Entries at 1.0 are left out, and an inverter list disagreeing with the aggregate SoC is scaled to it
    inverters = [_inverter(5.0, 10.0), _inverter(5.0, 10.0)]
    charge, discharge = derive_fleet_power_curves(inverters, 10.0)
    if any(abs(v - 1.0) <= 1e-4 for v in charge.values()) or any(abs(v - 1.0) <= 1e-4 for v in discharge.values()):
        print("ERROR: Curves should leave out entries at 1.0")
        failed = True
    uneven = [_inverter(10.0, 10.0), _inverter(2.0, 10.0)]
    base_charge, base_discharge = derive_fleet_power_curves(uneven, 12.0)
    scaled_charge, scaled_discharge = derive_fleet_power_curves(uneven, 6.0)
    if scaled_discharge == base_discharge and scaled_charge == base_charge:
        print("ERROR: A different aggregate SoC should change the derived curves of uneven inverters")
        failed = True

    # No rate, nothing to derive
    dead = [_inverter(5.0, 10.0, rate_kw=0.0), _inverter(5.0, 10.0, rate_kw=0.0)]
    charge, discharge = derive_fleet_power_curves(dead, 10.0)
    if charge is not None or discharge is not None:
        print("ERROR: Inverters with no rate should give no curves")
        failed = True

    return failed


def test_fleet_power_curves_prediction(my_predbat):
    """
    Test the derived fleet curves reach the plan: the Prediction's rate curves are untouched when the option is off or the
    inverters are balanced, and held under what the inverters can deliver when one has run down
    """
    failed = False
    step = PREDICT_STEP
    pv_step = {minute: 0.0 for minute in range(0, my_predbat.forecast_minutes, step)}
    load_step = {minute: 0.1 for minute in range(0, my_predbat.forecast_minutes, step)}
    rate = 2.6 / 60.0

    def fleet(socs):
        """Two inverters of 10 kWh at the given SoCs"""
        return [SimpleNamespace(soc_kw=soc, soc_max=10.0, reserve=0.4, battery_rate_max_charge=rate, battery_rate_max_discharge=rate) for soc in socs]

    saved = (my_predbat.inverters, my_predbat.soc_kw, my_predbat.soc_max, my_predbat.battery_rate_max_charge, my_predbat.battery_rate_max_discharge, getattr(my_predbat, "fleet_power_curves_enable", False))
    my_predbat.soc_max = 20.0
    my_predbat.battery_rate_max_charge = 2 * rate
    my_predbat.battery_rate_max_discharge = 2 * rate

    def curves(enable, socs):
        """The Prediction's (charge, discharge) rate curves for 10 degrees"""
        my_predbat.inverters = fleet(socs)
        my_predbat.soc_kw = sum(socs)
        my_predbat.fleet_power_curves_enable = enable
        return Prediction(my_predbat, pv_step, pv_step, load_step, load_step).rate_curves(10)

    off_charge, off_discharge = curves(False, [10.0, 0.4])
    balanced_charge, balanced_discharge = curves(True, [5.0, 5.0])
    plain_charge, plain_discharge = curves(False, [5.0, 5.0])
    # Only the ends differ for a balanced fleet: nothing can be delivered at reserve or taken at full, where the plain curve still shows the rate
    if balanced_charge[5:100] != plain_charge[5:100] or balanced_discharge[5:100] != plain_discharge[5:100]:
        print("ERROR: Balanced inverters should leave the rate curves unchanged between reserve and full")
        failed = True
    uneven_charge, uneven_discharge = curves(True, [10.0, 0.4])
    if uneven_discharge[10] > rate * 1.01 or uneven_discharge[10] <= 0:
        print("ERROR: With one inverter run down only one inverter's discharge rate should remain at 10%, got {} against {}".format(uneven_discharge[10], rate))
        failed = True
    if uneven_charge[75] > rate * 1.01:
        print("ERROR: With one inverter full only one inverter's charge rate should remain at 75%, got {} against {}".format(uneven_charge[75], rate))
        failed = True
    if off_charge != curves(False, [10.0, 0.4])[0] or off_discharge != curves(False, [10.0, 0.4])[1]:
        print("ERROR: The option off should not change the curves")
        failed = True
    if len(uneven_discharge) != len(plain_discharge) or uneven_discharge[101:] != plain_discharge[101:]:
        print("ERROR: The curve's rated rate and temperature cap entries should be kept")
        failed = True

    (my_predbat.inverters, my_predbat.soc_kw, my_predbat.soc_max, my_predbat.battery_rate_max_charge, my_predbat.battery_rate_max_discharge, my_predbat.fleet_power_curves_enable) = saved
    return failed
