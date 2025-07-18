# -----------------------------------------------------------------------------
# Predbat Home Battery System
# Copyright Trefor Southwell 2024 - All Rights Reserved
# This application maybe used for personal use only and not for commercial use
# -----------------------------------------------------------------------------
# fmt off
# pylint: disable=consider-using-f-string
# pylint: disable=line-too-long
# pylint: disable=attribute-defined-outside-init

from datetime import timedelta
from config import MINUTE_WATT
from utils import dp0, dp2, dp3, calc_percent_limit, find_charge_rate
from inverter import Inverter

"""
Execute Predbat plan
"""


class Execute:
    def execute_plan(self):
        status_extra = ""

        in_alert = self.alert_active_keep.get(self.minutes_now, 0)

        if self.holiday_days_left > 0:
            status = "Demand (Holiday)"
        else:
            status = "Demand"

        if self.inverter_needs_reset:
            self.reset_inverter()

        isCharging = False
        isExporting = False
        for inverter in self.inverters:
            if inverter.id not in self.count_inverter_writes:
                self.count_inverter_writes[inverter.id] = 0

            # Read-only mode
            if self.set_read_only:
                status = "Read-Only"
                continue
            # Inverter is in calibration mode
            if inverter.in_calibration:
                status = "Calibration"
                self.log("Inverter {} is in calibration mode, not executing plan and enabling charge/discharge at full rate.".format(inverter.id))
                for inverter in self.inverters:
                    inverter.adjust_charge_rate(inverter.battery_rate_max_charge * MINUTE_WATT)
                    inverter.adjust_discharge_rate(inverter.battery_rate_max_discharge * MINUTE_WATT)
                    inverter.adjust_battery_target(100.0, False)
                    inverter.adjust_reserve(0)
                break

            resetDischarge = self.set_charge_window or self.set_export_window
            resetCharge = self.set_charge_window or self.set_export_window
            resetPause = self.set_charge_window or self.set_export_window
            resetReserve = self.set_charge_window or self.set_export_window
            disabled_charge_window = False
            disabled_export = False

            # Re-programme charge window based on low rates?
            if self.set_charge_window and self.charge_window_best:
                # Find the next best window and save it
                window = self.charge_window_best[0]
                minutes_start = window["start"]
                minutes_end = window["end"]

                # Combine contiguous windows
                for windows in self.charge_window_best:
                    if minutes_end == windows["start"]:
                        minutes_end = windows["end"]
                        if self.debug_enable:
                            self.log("Combine window with next window {}-{}".format(self.time_abs_str(windows["start"]), self.time_abs_str(windows["end"])))

                # Avoid adjust avoid start time forward when it's already started
                if (inverter.charge_start_time_minutes <= self.minutes_now) and (self.minutes_now >= minutes_start):
                    self.log("Include original charge start {}, keeping this instead of new start {}".format(self.time_abs_str(inverter.charge_start_time_minutes), self.time_abs_str(minutes_start)))
                    minutes_start = inverter.charge_start_time_minutes

                # Avoid having too long a period to configure as registers only support 24-hours
                if (minutes_start < self.minutes_now) and ((minutes_end - minutes_start) >= 24 * 60):
                    minutes_start = int(self.minutes_now / 30) * 30
                    self.log("Move on charge window start time to avoid wrap - new start {}".format(self.time_abs_str(minutes_start)))

                # Span midnight allowed?
                if not inverter.inv_can_span_midnight:
                    if minutes_start < 24 * 60 and minutes_end >= 24 * 60:
                        minutes_end = 24 * 60 - 1

                # Are we currently in the export window?
                inExportWindow = False
                if self.set_export_window and self.export_window_best:
                    if self.minutes_now >= self.export_window_best[0]["start"] and self.minutes_now < self.export_window_best[0]["end"]:
                        inExportWindow = True

                # Check if start is within 24 hours of now and end is in the future
                if (not inExportWindow) and ((minutes_start - self.minutes_now) < (24 * 60)) and (minutes_end > self.minutes_now):
                    charge_start_time = self.midnight_utc + timedelta(minutes=minutes_start)
                    charge_end_time = self.midnight_utc + timedelta(minutes=minutes_end)
                    self.log("Inverter {} Charge window will be: {} - {} - current soc {} target {}".format(inverter.id, charge_start_time, charge_end_time, inverter.soc_percent, self.charge_limit_percent_best[0]))
                    # Are we actually charging?
                    if self.minutes_now >= minutes_start and self.minutes_now < minutes_end:
                        target_soc = self.charge_limit_percent_best[0] if self.charge_limit_best != self.reserve else self.soc_kw
                        inv_target_soc = self.adjust_battery_target_multi(inverter, target_soc, True, False, check=True)

                        current_charge_rate = inverter.get_current_charge_rate()
                        new_charge_rate, new_charge_rate_real = find_charge_rate(
                            self.minutes_now,
                            inverter.soc_kw,
                            window,
                            inv_target_soc * inverter.soc_max / 100.0,
                            inverter.battery_rate_max_charge,
                            inverter.soc_max,
                            self.battery_charge_power_curve,
                            self.set_charge_low_power,
                            self.charge_low_power_margin,
                            self.battery_rate_min,
                            self.battery_rate_max_scaling,
                            self.battery_loss,
                            self.log,
                            inverter.battery_temperature,
                            self.battery_temperature_charge_curve,
                            current_charge_rate=current_charge_rate / MINUTE_WATT,
                        )
                        new_charge_rate = int(new_charge_rate * MINUTE_WATT)

                        self.log(
                            "Inverter {} Target SOC {} (this inverter {}) Battery temperature {} Select charge rate {}w (real {}w) current charge rate {}".format(
                                inverter.id, target_soc, inv_target_soc, inverter.battery_temperature, new_charge_rate, new_charge_rate_real * MINUTE_WATT, current_charge_rate
                            )
                        )

                        # Adjust charge rate if we are more than 10% out or we are going back to Max charge rate
                        max_rate = inverter.battery_rate_max_charge * MINUTE_WATT
                        if abs(new_charge_rate - current_charge_rate) > (0.1 * max_rate) or (new_charge_rate == max_rate):
                            inverter.adjust_charge_rate(new_charge_rate)
                        resetCharge = False

                        if inverter.inv_charge_discharge_with_rate:
                            inverter.adjust_discharge_rate(0)
                            resetDischarge = False

                        # Can only freeze charge if all inverters have an SOC above the reserve
                        can_freeze_charge = True
                        for check in self.inverters:
                            if check.soc_kw < inverter.reserve:
                                can_freeze_charge = False
                                break
                            if not check.inv_has_timed_pause and (check.reserve_max < check.soc_percent):
                                can_freeze_charge = False
                                break
                        if (self.charge_limit_best[0] == self.reserve) and self.soc_kw >= self.reserve and can_freeze_charge:
                            if self.set_soc_enable and ((self.set_reserve_enable and self.set_reserve_hold and inverter.reserve_max >= inverter.soc_percent) or inverter.inv_has_timed_pause):
                                inverter.disable_charge_window()
                                disabled_charge_window = True
                                if self.set_reserve_enable and (not inverter.inv_has_timed_pause):
                                    inverter.adjust_reserve(min(inverter.soc_percent + 1, 100))
                                    resetReserve = False
                            else:
                                inverter.adjust_charge_window(charge_start_time, charge_end_time, self.minutes_now)

                            if inverter.inv_has_timed_pause:
                                inverter.adjust_pause_mode(pause_discharge=True)
                                resetPause = False
                            else:
                                inverter.adjust_discharge_rate(0)
                                resetDischarge = False

                            status = "Freeze charging"
                            status_extra = " target {}%".format(inverter.soc_percent)
                            self.log("Inverter {} Freeze charging with soc {}%".format(inverter.id, inverter.soc_percent))
                        else:
                            # We can only hold charge if a) we have a way to hold the charge level on the reserve or with a pause feature
                            # and the current charge level is above the target for all inverters
                            target_soc = calc_percent_limit(max(self.charge_limit_best[0] if self.charge_limit_best[0] != self.reserve else self.soc_kw, self.reserve), self.soc_max)
                            if self.set_soc_enable and self.soc_percent >= target_soc:
                                status = "Hold charging"
                                self.log("Inverter {} Hold charging as soc {}% is above target {}% set_discharge_during_charge {}".format(inverter.id, self.soc_percent, target_soc, self.charge_limit_percent_best[0], self.set_discharge_during_charge))

                                if (target_soc < 100.0) and (abs(inverter.soc_percent - target_soc) <= 1.0):
                                    # If we are within 1% of the target but not at 100% then we can hold charge
                                    # otherwise keep charging enabled
                                    if self.set_soc_enable and ((self.set_reserve_enable and self.set_reserve_hold and inverter.reserve_max >= inverter.soc_percent) or inverter.inv_has_timed_pause):
                                        inverter.disable_charge_window()
                                        disabled_charge_window = True

                                        if self.set_reserve_enable and not inverter.inv_has_timed_pause:
                                            inverter.adjust_reserve(min(inverter.soc_percent + 1, 100))
                                            resetReserve = False
                                    else:
                                        inverter.adjust_charge_window(charge_start_time, charge_end_time, self.minutes_now)

                                    # Pause?
                                    if inverter.inv_has_timed_pause:
                                        inverter.adjust_pause_mode(pause_discharge=True)
                                        resetPause = False
                                    else:
                                        inverter.adjust_discharge_rate(0)
                                        resetDischarge = False
                                else:
                                    inverter.adjust_charge_window(charge_start_time, charge_end_time, self.minutes_now)

                            else:
                                status = "Charging"
                                inverter.adjust_charge_window(charge_start_time, charge_end_time, self.minutes_now)

                            status_extra = " target {}%-{}%".format(inverter.soc_percent, target_soc)

                        if not self.set_discharge_during_charge and resetPause:
                            # Do we discharge discharge during charge
                            if inverter.inv_has_timed_pause:
                                inverter.adjust_pause_mode(pause_discharge=True)
                                resetPause = False
                            else:
                                inverter.adjust_discharge_rate(0)
                                resetDischarge = False
                            self.log("Disabling discharge during charge due to set_discharge_during_charge being False")

                        isCharging = True
                        self.isCharging_Target = self.charge_limit_best[0]
                    else:
                        # Configure the charge window start/end times if in the time window to set them
                        if (self.minutes_now < minutes_end) and ((minutes_start - self.minutes_now) <= self.set_window_minutes):
                            # We must re-program if we are about to start a new charge window or the currently configured window is about to start or has started
                            # If we are going into freeze mode but haven't yet then don't configure the charge window as it will mean a spike of charging first
                            if not isCharging and self.charge_limit_best[0] == self.reserve:
                                self.log("Charge window will be disabled as freeze charging is planned")
                                inverter.disable_charge_window()
                            else:
                                self.log(
                                    "Inverter {} configuring charge window now (now {} target set_window_minutes {} charge start time {}".format(inverter.id, self.time_abs_str(self.minutes_now), self.set_window_minutes, self.time_abs_str(minutes_start))
                                )
                                inverter.adjust_charge_window(charge_start_time, charge_end_time, self.minutes_now)
                        else:
                            self.log(
                                "Inverter {} disabled charge window while waiting for schedule (now {} target set_window_minutes {} charge start time {})".format(
                                    inverter.id, self.time_abs_str(self.minutes_now), self.set_window_minutes, self.time_abs_str(minutes_start)
                                )
                            )
                            inverter.disable_charge_window()

                    # Set configured window minutes for the SoC adjustment routine
                    inverter.charge_start_time_minutes = minutes_start
                    inverter.charge_end_time_minutes = minutes_end
                else:
                    self.log(
                        "Inverter {} Disabled charge window while waiting for schedule (now {} target set_window_minutes {} charge start time {})".format(
                            inverter.id, self.time_abs_str(self.minutes_now), self.set_window_minutes, self.time_abs_str(minutes_start)
                        )
                    )
                    inverter.disable_charge_window()
            elif self.set_charge_window:
                self.log("Inverter {} No charge window yet, waiting for schedule.".format(inverter.id))
                inverter.disable_charge_window()
            else:
                self.log("Inverter {} Set charge window is disabled".format(inverter.id))

            # Set discharge modes/window?
            if self.set_export_window and self.export_window_best:
                window = self.export_window_best[0]
                minutes_start = window["start"]
                minutes_end = window["end"]

                # Avoid adjust avoid start time forward when it's already started
                if (inverter.discharge_start_time_minutes <= self.minutes_now) and (self.minutes_now >= minutes_start):
                    minutes_start = inverter.discharge_start_time_minutes
                    # Don't allow overlap with charge window
                    if minutes_start < inverter.charge_end_time_minutes and minutes_end >= inverter.charge_start_time_minutes:
                        minutes_start = max(window["start"], self.minutes_now)
                    else:
                        self.log(
                            "Include original export start {} with our start which is {} (charge start {} end {})".format(
                                self.time_abs_str(inverter.discharge_start_time_minutes),
                                self.time_abs_str(minutes_start),
                                self.time_abs_str(inverter.charge_start_time_minutes),
                                self.time_abs_str(inverter.charge_end_time_minutes),
                            )
                        )

                # Avoid having too long a period to configure as registers only support 24-hours
                if (minutes_start < self.minutes_now) and ((minutes_end - minutes_start) >= 24 * 60):
                    minutes_start = int(self.minutes_now / 30) * 30
                    self.log("Move on export window start time to avoid wrap - new start {}".format(self.time_abs_str(minutes_start)))

                export_adjust = 1
                # Span midnight allowed?
                if not inverter.inv_can_span_midnight:
                    if minutes_start < 24 * 60 and minutes_end >= 24 * 60:
                        minutes_end = 24 * 60 - 1
                    export_adjust = 0

                # Overlap into charge slot if 1 minute was added, then don't add the 1 minute
                if inverter.charge_start_time_minutes == minutes_end:
                    export_adjust = 0

                # Turn minutes into time
                discharge_start_time = self.midnight_utc + timedelta(minutes=minutes_start)
                discharge_end_time = self.midnight_utc + timedelta(minutes=(minutes_end + export_adjust))  # Add in 1 minute margin to allow Predbat to restore demand mode
                discharge_soc = max((int(self.export_limits_best[0]) * self.soc_max) / 100.0, self.reserve, self.best_soc_min)
                self.log("Next export window will be: {} - {} at reserve {}".format(discharge_start_time, discharge_end_time, self.export_limits_best[0]))
                if (self.minutes_now >= minutes_start) and (self.minutes_now < minutes_end) and (self.export_limits_best[0] < 100.0):
                    if not self.set_export_freeze_only and self.export_limits_best[0] < 99.0 and (self.soc_kw > discharge_soc):
                        if self.set_export_low_power:
                            export_rate_adjust = 1 - (self.export_limits_best[0] - int(self.export_limits_best[0]))
                        else:
                            export_rate_adjust = 1.0

                        self.log("Exporting now - current SoC {} and target {} power adjust {}".format(self.soc_kw, dp2(discharge_soc), export_rate_adjust))

                        inverter.adjust_discharge_rate(inverter.battery_rate_max_discharge * export_rate_adjust * MINUTE_WATT)
                        resetDischarge = False
                        inverter.adjust_force_export(True, discharge_start_time, discharge_end_time)
                        if inverter.inv_charge_discharge_with_rate:
                            inverter.adjust_charge_rate(0)
                            resetCharge = False
                        isExporting = True
                        target = self.export_window_best[0].get("target", self.export_limits_best[0])
                        self.isExporting_Target = int(target)

                        status = "Exporting"
                        status_extra = " target {}%-{}%".format(inverter.soc_percent, int(target))
                        # Immediate export mode
                    else:
                        inverter.adjust_force_export(False)
                        disabled_export = True
                        if self.set_export_freeze and self.export_limits_best[0] == 99:
                            # In export freeze mode we disable charging during export slots
                            if inverter.inv_charge_discharge_with_rate:
                                inverter.adjust_charge_rate(0)
                                resetCharge = False
                            if inverter.inv_has_timed_pause:
                                inverter.adjust_pause_mode(pause_charge=True)
                                resetPause = False
                            else:
                                inverter.adjust_charge_rate(0)
                                resetCharge = False
                            self.log("Export Freeze as exporting is now at/below target - current SoC {} and target {}".format(self.soc_kw, discharge_soc))
                            status = "Freeze exporting"
                            status_extra = " current SoC {}%".format(inverter.soc_percent)  # Discharge limit (99) is meaningless when Freeze Exporting so don't display it
                            isExporting = True
                            target = self.export_window_best[0].get("target", self.export_limits_best[0])
                            self.isExporting_Target = int(target)
                        else:
                            status = "Hold exporting"
                            target = self.export_window_best[0].get("target", self.export_limits_best[0])
                            status_extra = " target {}%-{}%".format(inverter.soc_percent, int(target))
                            self.log("Export Hold (Demand mode) as export is now at/below target or freeze only is set - current SoC {} and target {}".format(self.soc_kw, discharge_soc))
                else:
                    if (self.minutes_now < minutes_end) and ((minutes_start - self.minutes_now) <= self.set_window_minutes) and (self.export_limits_best[0] < 100):
                        inverter.adjust_force_export(False, discharge_start_time, discharge_end_time)
                    else:
                        self.log("Not setting export as we are not yet within the export window - next time is {} - {}".format(self.time_abs_str(minutes_start), self.time_abs_str(minutes_end)))
                        inverter.adjust_force_export(False)
            elif self.set_export_window:
                self.log("No export window planned")
                inverter.adjust_force_export(False)

            # Car charging from battery disable?
            carHolding = False
            if self.set_charge_window and not self.car_charging_from_battery:
                for car_n in range(self.num_cars):
                    if self.car_charging_slots[car_n]:
                        window = self.car_charging_slots[car_n][0]
                        if self.car_charging_soc[car_n] >= self.car_charging_limit[car_n]:
                            self.log("Car {} is already charged, ignoring additional charging slot from {} - {}".format(car_n, self.time_abs_str(window["start"]), self.time_abs_str(window["end"])))
                        elif self.minutes_now >= window["start"] and self.minutes_now < window["end"]:
                            self.log("Car charging from battery is off, next slot for car {} is {} - {}".format(car_n, self.time_abs_str(window["start"]), self.time_abs_str(window["end"])))
                            # Don't disable discharge during force charge/discharge slots but otherwise turn it off to prevent
                            # from draining the battery
                            if not isExporting:
                                if inverter.inv_has_timed_pause:
                                    if resetPause:
                                        inverter.adjust_pause_mode(pause_discharge=True)
                                        resetPause = False
                                else:
                                    if resetDischarge:
                                        inverter.adjust_discharge_rate(0)
                                        resetDischarge = False
                                carHolding = True
                                self.log("Disabling battery discharge while the car {} is charging".format(car_n))
                                if "Hold for car" not in status:
                                    if status != "Demand":
                                        status += ", Hold for car"
                                    else:
                                        status = "Hold for car"
                            break

            # Iboost running?
            boostHolding = False
            if self.set_charge_window and self.iboost_enable and self.iboost_prevent_discharge and self.iboost_running_full and status not in ["Exporting", "Charging"]:
                if inverter.inv_has_timed_pause:
                    inverter.adjust_pause_mode(pause_discharge=True)
                    resetPause = False
                else:
                    inverter.adjust_discharge_rate(0)
                    resetDischarge = False
                boostHolding = True
                self.log("Disabling battery discharge while iBoost is running")
                if "Hold for iBoost" not in status:
                    if status != "Demand":
                        status += ", Hold for iBoost"
                    else:
                        status = "Hold for iBoost"

            # Reset charge/discharge rate
            if resetPause:
                inverter.adjust_pause_mode()
            if resetDischarge:
                inverter.adjust_discharge_rate(inverter.battery_rate_max_discharge * MINUTE_WATT)
            if resetCharge:
                inverter.adjust_charge_rate(inverter.battery_rate_max_charge * MINUTE_WATT)

            if self.charge_limit_best:
                clm = self.charge_limit_best[0]
            else:
                clm = 0

            # Set the SoC just before or within the charge window
            if self.set_soc_enable:
                if isExporting:
                    if not disabled_export and not self.set_reserve_enable:
                        # If we are discharging and not setting reserve then we should reset the target SoC to the discharge target
                        # as some inverters can use this as a target for discharge
                        self.adjust_battery_target_multi(inverter, int(self.export_limits_best[0]), isCharging, isExporting)
                    elif not inverter.inv_has_discharge_enable_time:
                        self.adjust_battery_target_multi(inverter, 0, isCharging, isExporting)
                    elif not self.inverter_hybrid and self.inverter_soc_reset and inverter.inv_has_target_soc:
                        # AC Coupled, charge to 100 on solar
                        self.log("Resetting charging SoC to 100 as we are not charging and inverter_soc_reset is enabled")
                        self.adjust_battery_target_multi(inverter, 100.0, isCharging, isExporting)
                    else:
                        # Reset to 0
                        self.adjust_battery_target_multi(inverter, 0, isCharging, isExporting)

                    # Immediate controls
                    if self.set_export_freeze and self.export_limits_best[0] == 99:
                        inverter.adjust_export_immediate(inverter.soc_percent, freeze=True)
                    elif not disabled_export:
                        inverter.adjust_export_immediate(int(self.export_limits_best[0]))
                    else:
                        inverter.adjust_export_immediate(0)

                elif self.charge_limit_best and (self.minutes_now < inverter.charge_end_time_minutes) and ((inverter.charge_start_time_minutes - self.minutes_now) <= self.set_soc_minutes) and not (disabled_charge_window):
                    if inverter.inv_has_charge_enable_time or isCharging:
                        # In charge freeze hold the target SoC at the current value
                        if (self.charge_limit_best[0] == self.reserve) and (inverter.soc_kw >= inverter.reserve):
                            if isCharging:
                                self.log("Within charge freeze setting target soc to current soc {}".format(inverter.soc_percent))
                                self.adjust_battery_target_multi(inverter, inverter.soc_percent, isCharging, isExporting, isFreezeCharge=True)
                                inverter.adjust_charge_immediate(inverter.soc_percent, freeze=True)
                            elif not inverter.inv_has_target_soc:
                                self.adjust_battery_target_multi(inverter, 0, isCharging, isExporting)
                            else:
                                # Not yet in the freeze, hold at 100% target SoC
                                self.log("Not yet in charge freeze, holding target soc at 100%")
                                self.adjust_battery_target_multi(inverter, 100.0, isCharging, isExporting)
                        else:
                            # If not charging and not hybrid we should reset the target % to 100 to avoid losing solar
                            if not self.inverter_hybrid and self.inverter_soc_reset and not isCharging and inverter.inv_has_target_soc:
                                self.log("Resetting charging SOC as we are not charging and inverter_soc_reset is enabled")
                                self.adjust_battery_target_multi(inverter, 100.0, isCharging, isExporting)
                            elif isCharging:
                                target_soc = calc_percent_limit(max(self.charge_limit_best[0] if self.charge_limit_best[0] != self.reserve else self.soc_kw, self.reserve), self.soc_max)
                                self.log("Setting charging SOC to {} as per target".format(target_soc))
                                self.adjust_battery_target_multi(inverter, target_soc, isCharging, isExporting)
                                if self.charge_limit_best[0] == self.reserve:
                                    inverter.adjust_charge_immediate(calc_percent_limit(max(inverter.soc_kw, inverter.reserve), inverter.soc_max), freeze=True)
                                else:
                                    inverter.adjust_charge_immediate(target_soc)
                            elif not inverter.inv_has_target_soc:
                                self.log("Setting charging SOC to 0 as we are not charging and inverter doesn't support target soc")
                                self.adjust_battery_target_multi(inverter, 0, isCharging, isExporting)
                            else:
                                target_soc = calc_percent_limit(max(self.charge_limit_best[0] if self.charge_limit_best[0] != self.reserve else self.soc_kw, self.reserve), self.soc_max)
                                self.log("Setting charging SOC to {} as per target for when charge window starts".format(target_soc))
                                self.adjust_battery_target_multi(inverter, target_soc, isCharging, isExporting)
                    else:
                        if not inverter.inv_has_target_soc:
                            # If the inverter doesn't support target soc and soc_enable is on then do that logic here:
                            if not isCharging and not isExporting:
                                self.log("Setting charging SOC to 0 as we are not charging or exporting and inverter doesn't support target soc")
                                self.adjust_battery_target_multi(inverter, 0, isCharging, isExporting)
                        elif not self.inverter_hybrid and self.inverter_soc_reset and inverter.inv_has_target_soc:
                            # AC Coupled, charge to 100 on solar
                            self.log("Resetting charging SoC to 100 as we are not charging and inverter_soc_reset is enabled")
                            self.adjust_battery_target_multi(inverter, 100.0, isCharging, isExporting)
                        else:
                            # Hybrid, no charge timer, set target soc back to 0
                            self.log("Setting charging SOC to 0 as we are not charging and the inverter doesn't support charge enable time")
                            self.adjust_battery_target_multi(inverter, 0, isCharging, isExporting)
                else:
                    if not inverter.inv_has_target_soc:
                        self.adjust_battery_target_multi(inverter, 0, isCharging, isExporting)
                        self.log("Setting charging SOC to 0 as we are not within the charge window and inverter doesn't support target soc")
                    elif not self.inverter_hybrid and self.inverter_soc_reset:
                        self.log(
                            "Resetting charging SoC as we are not within the window or charge is disabled and inverter_soc_reset is enabled (now {} target set_soc_minutes {} charge start time {})".format(
                                self.time_abs_str(self.minutes_now), self.set_soc_minutes, self.time_abs_str(inverter.charge_start_time_minutes)
                            )
                        )
                        self.adjust_battery_target_multi(inverter, 100.0, isCharging, isExporting)
                    else:
                        self.log(
                            "Not setting charging SoC as we are not within the window (now {} target set_soc_minutes {} charge start time {})".format(
                                self.time_abs_str(self.minutes_now), self.set_soc_minutes, self.time_abs_str(inverter.charge_start_time_minutes)
                            )
                        )
                        if not inverter.inv_has_charge_enable_time:
                            self.adjust_battery_target_multi(inverter, 0, isCharging, isExporting)

                    # Charge immediate
                    if isCharging:
                        if self.charge_limit_best[0] == self.reserve:
                            inverter.adjust_charge_immediate(inverter.soc_percent, freeze=True)
                        else:
                            inverter.adjust_charge_immediate(calc_percent_limit(max(self.charge_limit_best[0], self.reserve), self.soc_max), freeze=True)

            # Charging/Discharging off via service
            if not isCharging and self.set_charge_window:
                if carHolding or boostHolding:
                    inverter.adjust_charge_immediate(inverter.soc_percent, freeze=True)
                else:
                    inverter.adjust_charge_immediate(0)
            if not isExporting and self.set_export_window:
                inverter.adjust_export_immediate(0)

            # Reset reserve as discharge is enable but not running right now
            if self.set_reserve_enable and resetReserve:
                inverter.adjust_reserve(0)

            # Count register writes
            self.log("Inverter {} count register writes {}".format(inverter.id, inverter.count_register_writes))
            self.count_inverter_writes[inverter.id] += inverter.count_register_writes
            inverter.count_register_writes = 0

        # Set the charge/discharge status information
        self.set_charge_export_status(isCharging and not disabled_charge_window, isExporting and not disabled_export, not (isCharging or isExporting))
        self.isCharging = isCharging
        self.isExporting = isExporting

        if in_alert > 0:
            status += " [Alert]"

        return status, status_extra

    def adjust_battery_target_multi(self, inverter, soc, is_charging, is_exporting, isFreezeCharge=False, check=False):
        """
        Adjust target SoC based on the current SoC of all the inverters accounting for their
        charge rates and battery capacities
        """
        target_kwh = dp2(self.soc_max * (soc / 100.0))
        soc_percent = calc_percent_limit(self.soc_kw, self.soc_max)

        if isFreezeCharge:
            new_soc_percent = soc
            if not check:
                self.log("Inverter {} adjust target soc for hold to {}% based on requested all inverter soc {}%".format(inverter.id, new_soc_percent, soc))
        elif soc == 100.0:
            new_soc_percent = 100.0
            if not check:
                self.log("Inverter {} adjust target soc for charge to {}% based on requested all inverter soc {}%".format(inverter.id, new_soc_percent, soc))
        elif soc == 0.0:
            new_soc_percent = 0.0
            if not check:
                self.log("Inverter {} adjust target soc for export to {}% based on requested all inverter soc {}%".format(inverter.id, new_soc_percent, soc))
        else:
            add_kwh = target_kwh - self.soc_kw
            add_this = add_kwh * (inverter.battery_rate_max_charge / self.battery_rate_max_charge)
            new_soc_kwh = max(min(inverter.soc_kw + add_this, inverter.soc_max), inverter.reserve)
            new_soc_percent = calc_percent_limit(new_soc_kwh, inverter.soc_max)
            if not check:
                self.log(
                    "Inverter {} adjust target soc for charge to {}% ({}kWh/{}kWh {}kWh) based on going from {}% -> {}% total add is {}kWh and this battery needs to add {}kWh to get to {}kWh".format(
                        inverter.id, soc, target_kwh, self.soc_max, inverter.soc_max, soc_percent, new_soc_percent, dp2(add_kwh), dp2(add_this), dp2(new_soc_kwh)
                    )
                )
        if not check:
            inverter.adjust_battery_target(new_soc_percent, is_charging, is_exporting)
        return new_soc_percent

    def reset_inverter(self):
        """
        Reset inverter to safe mode
        """
        if not self.set_read_only or (self.inverter_needs_reset_force in ["set_read_only"]):
            self.last_service_hash = {}
            # Don't reset in read only mode unless forced
            for inverter in self.inverters:
                self.log("Reset inverter settings to safe mode (set_charge_window={} set_export_window={} force={})".format(self.set_charge_window, self.set_export_window, self.inverter_needs_reset_force))
                if self.set_charge_window or (self.inverter_needs_reset_force in ["set_read_only", "mode"]):
                    inverter.adjust_charge_rate(inverter.battery_rate_max_charge * MINUTE_WATT)
                    inverter.disable_charge_window()
                    inverter.adjust_charge_immediate(0)
                    inverter.adjust_battery_target(100.0, False)
                    inverter.adjust_pause_mode()
                    self.isCharging = False
                if self.set_charge_window or self.set_export_window or (self.inverter_needs_reset_force in ["set_read_only", "mode"]):
                    inverter.adjust_reserve(0)
                if self.set_export_window or (self.inverter_needs_reset_force in ["set_read_only", "mode"]):
                    inverter.adjust_discharge_rate(inverter.battery_rate_max_discharge * MINUTE_WATT)
                    inverter.adjust_force_export(False)
                    inverter.adjust_export_immediate(0)
                    self.isExporting = False

        self.inverter_needs_reset = False
        self.inverter_needs_reset_force = ""

    def fetch_inverter_data(self, create=True):
        """
        Fetch data about the inverters
        """
        # Find the inverters
        self.num_inverters = int(self.get_arg("num_inverters", 1))
        self.inverter_limit = 0.0
        self.export_limit = 0.0
        self.charge_window = []
        self.export_window = []
        self.export_limits = []
        self.current_charge_limit_kwh = 0.0
        self.soc_kw = 0.0
        self.soc_max = 0.0
        self.reserve = 0.0
        self.reserve_percent = 0.0
        self.reserve_current = 0.0
        self.reserve_current_percent = 0.0
        self.battery_rate_max_charge = 0.0
        self.battery_rate_max_discharge = 0.0
        self.battery_rate_max_charge_scaled = 0.0
        self.battery_rate_max_discharge_scaled = 0.0
        self.battery_rate_min = 0
        self.charge_rate_now = 0.0
        self.discharge_rate_now = 0.0
        self.pv_power = 0
        self.load_power = 0
        self.battery_power = 0
        self.battery_temperature = 0
        self.grid_power = 0
        found_first = False

        if create:
            self.inverters = []

        # For each inverter get the details
        for id in range(self.num_inverters):
            if create:
                inverter = Inverter(self, id)
                self.inverters.append(inverter)
            else:
                inverter = self.inverters[id]
            inverter.update_status(self.minutes_now)

            if id == 0 and (not self.computed_charge_curve or self.battery_charge_power_curve_auto) and not self.battery_charge_power_curve:
                curve = inverter.find_charge_curve(discharge=False)
                if curve and self.battery_charge_power_curve_auto:
                    self.log("Saved computed battery charge power curve")
                    self.battery_charge_power_curve = curve
                self.computed_charge_curve = True
            if id == 0 and (not self.computed_discharge_curve or self.battery_discharge_power_curve_auto) and not self.battery_discharge_power_curve:
                curve = inverter.find_charge_curve(discharge=True)
                if curve and self.battery_discharge_power_curve_auto:
                    self.log("Saved computed battery discharge power curve")
                    self.battery_discharge_power_curve = curve
                self.computed_discharge_curve = True

            # As the inverters will run in lockstep, we will initially look at the programming of the first enabled one for the current window setting
            if not found_first:
                found_first = True
                self.charge_window = inverter.charge_window
                self.export_window = inverter.export_window
                self.export_limits = inverter.export_limits
                if not inverter.inv_support_discharge_freeze:
                    # Force off unsupported feature
                    self.log("Note: Inverter does not support discharge freeze - disabled")
                    self.set_export_freeze = False
                    self.set_export_freeze_only = False
                if not inverter.inv_support_charge_freeze:
                    # Force off unsupported feature
                    self.log("Note: Inverter does not support charge freeze - disabled")
                    self.set_charge_freeze = False
                if not inverter.inv_has_reserve_soc:
                    self.log("Note: Inverter does not support reserve - disabling reserve functions")
                    self.set_reserve_enable = False
                    self.set_reserve_hold = False
                    self.set_discharge_during_charge = True
            self.current_charge_limit_kwh += dp2(inverter.current_charge_limit * inverter.soc_max / 100.0)
            self.soc_max += inverter.soc_max
            self.soc_kw += inverter.soc_kw
            self.reserve += inverter.reserve
            self.reserve_current += inverter.reserve_current
            self.battery_rate_max_charge += inverter.battery_rate_max_charge
            self.battery_rate_max_discharge += inverter.battery_rate_max_discharge
            self.battery_rate_max_charge_scaled += inverter.battery_rate_max_charge_scaled
            self.battery_rate_max_discharge_scaled += inverter.battery_rate_max_discharge_scaled
            self.charge_rate_now += inverter.charge_rate_now
            self.discharge_rate_now += inverter.discharge_rate_now
            self.battery_rate_min += inverter.battery_rate_min
            self.inverter_limit += inverter.inverter_limit
            self.export_limit += inverter.export_limit
            self.pv_power += inverter.pv_power
            self.load_power += inverter.load_power
            self.battery_power += inverter.battery_power
            self.grid_power += inverter.grid_power
            self.current_charge_limit = calc_percent_limit(self.current_charge_limit_kwh, self.soc_max)
            self.battery_temperature += inverter.battery_temperature

        # Work out battery temperature
        self.battery_temperature = int(dp0(self.battery_temperature / self.num_inverters))

        # Remove extra decimals
        self.soc_max = dp3(self.soc_max)
        self.soc_kw = dp3(self.soc_kw)
        self.soc_percent = calc_percent_limit(self.soc_kw, self.soc_max)
        self.reserve = dp3(self.reserve)
        self.reserve_percent = calc_percent_limit(self.reserve, self.soc_max)
        self.reserve_current = dp3(self.reserve_current)
        self.reserve_current_percent = calc_percent_limit(self.reserve_current, self.soc_max)

        self.log(
            "Found {} inverters totals: min reserve {} current reserve {} soc_max {} soc {} charge rate {} kW discharge rate {} kW battery_rate_min {} w ac limit {} export limit {} kW loss charge {} % loss discharge {} % inverter loss {} %".format(
                len(self.inverters),
                self.reserve,
                self.reserve_current,
                self.soc_max,
                self.soc_kw,
                self.charge_rate_now * 60,
                self.discharge_rate_now * 60,
                self.battery_rate_min * MINUTE_WATT,
                dp3(self.inverter_limit * 60),
                dp3(self.export_limit * 60),
                100 - int(self.battery_loss * 100),
                100 - int(self.battery_loss_discharge * 100),
                100 - int(self.inverter_loss * 100),
            )
        )

        # Work out current charge limits and publish charge limit base
        self.charge_limit = [self.current_charge_limit * self.soc_max / 100.0 for i in range(len(self.charge_window))]
        self.charge_limit_percent = calc_percent_limit(self.charge_limit, self.soc_max)
        self.publish_charge_limit(self.charge_limit, self.charge_window, self.charge_limit_percent, best=False)

        self.log("Base charge window {}".format(self.window_as_text(self.charge_window, self.charge_limit_percent)))
        self.log("Base export window {}".format(self.window_as_text(self.export_window, self.export_limits)))

    def balance_inverters(self):
        """
        Attempt to balance multiple inverters
        """
        # Charge rate resets
        balance_reset_charge = {}
        balance_reset_discharge = {}

        self.log(
            "BALANCE: Enabled balance charge {} discharge {} crosscharge {} threshold charge {} discharge {}".format(
                self.balance_inverters_charge,
                self.balance_inverters_discharge,
                self.balance_inverters_crosscharge,
                self.balance_inverters_threshold_charge,
                self.balance_inverters_threshold_discharge,
            )
        )
        self.update_time(print=False)

        # For each inverter get the details
        num_inverters = int(self.get_arg("num_inverters", 1))

        inverters = []
        for id in range(num_inverters):
            inverter = Inverter(self, id, quiet=True)
            inverter.update_status(self.minutes_now, quiet=True)
            if inverter.in_calibration:
                self.log("Inverter {} is in calibration mode, not balancing".format(id))
                return
            inverters.append(inverter)

        out_of_balance = False  # Are all the SoC % the same
        total_battery_power = 0  # Total battery power across inverters
        total_max_rate = 0  # Total battery max rate across inverters
        total_charge_rates = 0  # Current total charge rates
        total_discharge_rates = 0  # Current total discharge rates
        total_pv_power = 0  # Current total PV power
        total_load_power = 0  # Current load power
        socs = []
        reserves = []
        battery_powers = []
        pv_powers = []
        battery_max_rates = []
        charge_rates = []
        discharge_rates = []
        load_powers = []
        for inverter in inverters:
            socs.append(inverter.soc_percent)
            reserves.append(inverter.reserve_current)
            if inverter.soc_percent != inverters[0].soc_percent:
                out_of_balance = True
            battery_powers.append(inverter.battery_power)
            pv_powers.append(inverter.pv_power)
            load_powers.append(inverter.load_power)
            total_battery_power += inverter.battery_power
            total_pv_power += inverter.pv_power
            total_load_power += inverter.load_power
            battery_max_rates.append(inverter.battery_rate_max_discharge * MINUTE_WATT)
            total_max_rate += inverter.battery_rate_max_discharge * MINUTE_WATT
            charge_rates.append(inverter.charge_rate_now * MINUTE_WATT)
            total_charge_rates += inverter.charge_rate_now * MINUTE_WATT
            discharge_rates.append(inverter.discharge_rate_now * MINUTE_WATT)
            total_discharge_rates += inverter.discharge_rate_now * MINUTE_WATT
        self.log(
            "BALANCE: socs {} reserves {} battery_powers {} total {} battery_max_rates {} charge_rates {} pv_power {} load_power {} total {} discharge_rates {} total {}".format(
                socs,
                reserves,
                battery_powers,
                total_battery_power,
                battery_max_rates,
                charge_rates,
                pv_powers,
                load_powers,
                total_charge_rates,
                discharge_rates,
                total_discharge_rates,
            )
        )

        # Are we discharging
        during_discharge = total_battery_power >= 0.0
        during_charge = total_battery_power < 0.0

        # Work out min and max socs
        soc_min = min(socs)
        soc_max = max(socs)

        # Work out which inverters have low and high Soc
        soc_low = []
        soc_high = []
        for inverter in inverters:
            soc_low.append(inverter.soc_percent < soc_max and (abs(inverter.soc_percent - soc_max) >= self.balance_inverters_discharge))
            soc_high.append(inverter.soc_percent > soc_min and (abs(inverter.soc_percent - soc_min) >= self.balance_inverters_charge))

        above_reserve = []  # Is the battery above reserve?
        below_full = []  # Is the battery below full?
        can_power_house = []  # Could this inverter power the house alone?
        can_store_pv = []  # Can store the PV for the house alone?
        power_enough_discharge = []  # Inverter drawing enough power to be worth balancing
        power_enough_charge = []  # Inverter drawing enough power to be worth balancing
        for id in range(num_inverters):
            above_reserve.append((socs[id] - reserves[id]) >= 4.0)
            below_full.append(socs[id] < 100.0)
            can_power_house.append((total_discharge_rates - discharge_rates[id] - 200) >= total_battery_power)
            can_store_pv.append(total_pv_power <= (total_charge_rates - charge_rates[id]))
            power_enough_discharge.append(battery_powers[id] >= 50.0)
            power_enough_charge.append(inverters[id].battery_power <= -50.0)

        self.log(
            "BALANCE: out_of_balance {} above_reserve {} below_full {} can_power_house {} can_store_pv {} power_enough_discharge {} power_enough_charge {} soc_low {} soc_high {}".format(
                out_of_balance, above_reserve, below_full, can_power_house, can_store_pv, power_enough_discharge, power_enough_charge, soc_low, soc_high
            )
        )
        for this_inverter in range(num_inverters):
            other_inverter = (this_inverter + 1) % num_inverters
            if (
                self.balance_inverters_discharge
                and total_discharge_rates > 0
                and out_of_balance
                and during_discharge
                and soc_low[this_inverter]
                and above_reserve[other_inverter]
                and can_power_house[this_inverter]
                and (power_enough_discharge[this_inverter] or discharge_rates[this_inverter] == 0)
            ):
                self.log("BALANCE: Inverter {} is out of balance low - during discharge, attempting to balance it using inverter {}".format(this_inverter, other_inverter))
                balance_reset_discharge[this_inverter] = True
                inverters[this_inverter].adjust_discharge_rate(0, notify=False)
            elif (
                self.balance_inverters_charge
                and total_charge_rates > 0
                and out_of_balance
                and during_charge
                and soc_high[this_inverter]
                and below_full[other_inverter]
                and can_store_pv[this_inverter]
                and (power_enough_charge[this_inverter] or charge_rates[this_inverter] == 0)
            ):
                self.log("BALANCE: Inverter {} is out of balance high - during charge, attempting to balance it".format(this_inverter))
                balance_reset_charge[this_inverter] = True
                inverters[this_inverter].adjust_charge_rate(0, notify=False)
            elif self.balance_inverters_crosscharge and during_discharge and total_discharge_rates > 0 and power_enough_charge[this_inverter]:
                self.log("BALANCE: Inverter {} is cross charging during discharge, attempting to balance it".format(this_inverter))
                if soc_low[this_inverter] and can_power_house[other_inverter]:
                    balance_reset_discharge[this_inverter] = True
                    inverters[this_inverter].adjust_discharge_rate(0, notify=False)
                else:
                    balance_reset_charge[this_inverter] = True
                    inverters[this_inverter].adjust_charge_rate(0, notify=False)
            elif self.balance_inverters_crosscharge and during_charge and total_charge_rates > 0 and power_enough_discharge[this_inverter]:
                self.log("BALANCE: Inverter {} is cross discharging during charge, attempting to balance it".format(this_inverter))
                balance_reset_discharge[this_inverter] = True
                inverters[this_inverter].adjust_discharge_rate(0, notify=False)

        for id in range(num_inverters):
            if not balance_reset_charge.get(id, False) and total_charge_rates != 0 and charge_rates[id] == 0:
                self.log("BALANCE: Inverter {} reset charge rate to {} now balanced".format(id, inverter.battery_rate_max_charge * MINUTE_WATT))
                inverters[id].adjust_charge_rate(inverter.battery_rate_max_charge * MINUTE_WATT, notify=False)
            if not balance_reset_discharge.get(id, False) and total_discharge_rates != 0 and discharge_rates[id] == 0:
                self.log("BALANCE: Inverter {} reset discharge rate to {} now balanced".format(id, inverter.battery_rate_max_discharge * MINUTE_WATT))
                inverters[id].adjust_discharge_rate(inverter.battery_rate_max_discharge * MINUTE_WATT, notify=False)

        self.log("BALANCE: Completed this run")
