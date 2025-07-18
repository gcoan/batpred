# Customisation

This document describes the Predbat configuration items in Home Assistant that you can modify to customise Predbat to fit your needs.

All of these settings are entities that can be configured directly in Home Assistant (unlike the '[apps.yaml](apps-yaml.md)' configuration items that have to be edited with a file editor).

See [Displaying output data](output-data.md)
for information on how to view and edit these entities within
Home Assistant.

## Saving and restoring Predbat settings

The selector **select.predbat_saverestore** can be used to save your current Predbat settings to a YAML file (kept in the directory `/config/predbat_save/`) and to
restore the settings from one of these files.

Selecting the selector option **save current** will cause the settings to be saved to a date/time-stamped file.
You can rename this file yourself in the Home Assistant filesystem to give it a more human-readable name or delete it if you no longer want to keep it.
This is normally best done in an SSH window or via a Samba mount.

Selecting the option **restore default** will put all your settings back to the Predbat defaults.
Before the restore, the current Predbat settings will be saved to the file **previous.yaml** - should you have made a mistake you can restore them quickly again.

Selecting any of the .yaml files you have created will restore your settings from this file.

![image](https://github.com/springfall2008/batpred/assets/48591903/209442c1-bd4d-4812-84e2-c5a81794bd1d)

## Predbat mode

The mode that Predbat operates in will change the operation, this can be configured with **select.predbat_mode** drop-down menu as follows:

- Monitor
- Control SOC Only
- Control charge
- Control charge & discharge

If the **switch.predbat_set_read_only** is set to True then this prevents Predbat from making modifications to the inverter settings (regardless of the configuration).
Predbat will continue making and updating its prediction plan every 5 minutes, but no inverter changes will be made.
This is useful if you want to override what Predbat is planning to do (e.g. your own automation), or whilst you are learning how Predbat works before turning it on 'in anger'.

_NOTE: Changing the Predbat mode or the read-only switch will cause Predbat to reset the inverter settings to default, this will disable
both charge and discharge, reset charge and discharge rates to full power and reset the reserve to the default setting_

![image](https://github.com/springfall2008/batpred/assets/48591903/43faa962-6b8a-495a-88f8-f762aa1d55b8)

### Predbat Monitor mode

In **Monitor** mode Predbat will not control or Plan any charging or discharging, inverter balancing will take place if enabled,
and the plan will show just what is expected based on the current inverter configuration alone.

### Predbat Control SOC only mode

In **Control SOC only** mode Predbat will adjust the target charge percentage (SOC target) according to the Best plan, but the charge
window will not be modified.

This mode can be useful if you just have one fixed charge slot per day and you only want Predbat to control the percentage the battery is charged based on solar generation
and predicted house load.

_CAUTION: You must manually set any charging required on the inverter and if the charge window is disabled then no charging will take place._

### Predbat Control charge mode

In **Control charge** mode Predbat will set the charge times and charge percentages according to the Best plan, charging can be enabled and
disabled by Predbat.
Predbat will set the inverter into Eco mode when required to enable the battery to support house load, but it will not plan any forced discharging of the battery for export purposes.

This mode can be useful if you don't have an export rate, you have a 'no export' limitation from your electricity supplier, or if you want to preserve the battery for home demand.

### Predbat Control charge & discharge mode

In **Control charge & discharge** mode Predbat will set both charge and force export (discharge) times and control charge and force export percentages.

If you have set the **switch.predbat_set_export_freeze_only** set to True then forced export won't occur but Predbat can force the export
of solar power to the grid when desired.

## Expert mode

Predbat has a toggle switch called **switch.predbat_expert_mode** which is set to Off by default for new installs (On by default for upgraded installs).
A lot of Predbat's more advanced configuration options will not be available unless expert mode is enabled.
It's recommended for new users to start without expert mode and then maybe enable it later once you become more confident with the tool.

## Performance related

By default Predbat controls the inverter and updates the plan every 5 minutes, this can however use a lot of CPU power
especially on more complex tariffs like Agile when run on lower power machines such as Raspberry PIs and some thin clients.

You can tweak **input_number.predbat_calculate_plan_every** (_expert mode_) to reduce the frequency of replanning while
keeping the inverter control in the 5-minute slots. E.g. a value of 10 or 15 minutes should also give good results.

If you have performance problems leave **switch.predbat_calculate_second_pass** (_expert mode_) turned Off as it's
quite CPU intensive and provides very little improvement for most systems.

You can enable **combine_charge_slots** and **combine_export_slots** to speed up planning.
Note: Combining export slots may prevent optimal forced export. Combining charge slots is usually fine for tariffs with
longer periods of fixed rates but can limit the planning ability in some cases.

The number of threads you use can change your performance, you can set **threads** in `apps.yaml` to 0 to disable threading
if you don't have multiple CPUs available or set it to 'auto' (the default) to use one thread per CPU. It is recommended you don't set this to an odd number of threads.

## Battery loss options

**input_number.predbat_battery_loss** is an assumed percentage figure for energy lost when charging the battery, the default 0.05 is 5%.

**input_number.predbat_battery_loss_discharge** is an assumed percentage figure for energy lost whilst discharging the battery, the default 0.05 is 5%.

**input_number.predbat_inverter_loss** is an assumed percentage figure for  energy lost during the conversion within the inverter from DC to AC or AC to DC,
the default is 0% for legacy reasons but please adjust.

_TIP:_ Make sure you set the losses correctly, they are decimal percentages, so don't set them to '4' thinking it'll be 4%, Predbat will take this as being 400% and your plan will be very strange!

**switch.predbat_inverter_hybrid** Set to True if you have a hybrid inverter so no inverter losses will be applied for DC charging from Solar generation.
Set to False if you have an AC-coupled battery and inverter losses will be applied when charging from solar.
NB: This switch only applies when Predbat is modelling solar charging.
All grid charging (regardless of inverter type) has to undergo an AC to DC conversion so the inverter_loss % will be included in Predbat's model when charging from the grid.

**input_number.predbat_metric_battery_cycle** (_expert mode_) This sets a 'virtual cost' in pence per kWh on using your battery for charging and discharging.
Higher numbers will reduce battery cycles at the expense of using higher energy costs.<BR>
In theory, if you have a 9.5kWh battery and think it will last say 6000 complete cycles and it cost you £4000, then each full charge and discharge cycle is 19kWh
and so the cost per complete cycle is £4000 / 19 / 6000 = 3.5p.

Taking the 3.5p per cycle example, if you set predbat_metric_battery_cycle to 1.75 (half of 3.5) then Predbat will apply the "virtual cost" of 1.75p
to every kWh of charge and discharge of the battery.
This cost will be included in Predbat's cost optimisation plan when it decides whether to charge, discharge the battery or let the house run on grid import.<BR>
_NB: For clarity and to re-emphasise, the "virtual cost" will be applied to BOTH the cost calculation for charging AND for discharging the battery._

If you configure this number higher then more expensive plans will be selected which avoids charging and discharging your battery as much.

Note that the cycle cost will not be included in the cost predictions that Predbat produces such as the Predbat HTML plan or Apex charts,
it's just a cost taken into account by Predbat at the planning stage when the plan is calculated.<BR>
_NB: Setting this to a non-zero value will increase your daily cost, but will reduce your home battery usage._

Figures of around 0p-2p are recommended, the default is 0p per kWh.

**input_number.predbat_metric_battery_value_scaling** (_expert mode_) A percentage value that can be used to scale the value of the energy in the battery at the end of the plan.
The battery value is accounted for in the optimisations at the lowest future import rate including charging and inverter losses.
A value of 1.0 means no change to this, while lower than 1.0 means to value future battery levels less,
greater than 1.0 will value it more (and hence hold more charge at the end of the plan).

**input_number.metric_self_sufficiency** (_expert mode_) A price in pence per kWh used to skew the calculations towards self-sufficiency. Effectively saying to Predbat to
account for imports at a higher price than reality in the calculation and thus selecting plans with less import. If you want to be as self-sufficient as possible then set
this to the difference between your lowest import rate and the highest export rate to take exports that require additional import appear unprofitable. This setting will not
impact the real calculated costs and is only used for plan selection. Values of 5-10p may be worth trying if you prefer to avoid importing even if it saves you money.

## Scaling and weight options

**input_number.predbat_battery_rate_max_scaling** is a percentage factor to adjust your maximum charge rate from that reported by the inverter.
For example, a value of 0.95 would be 95% and indicate charging at 5% slower than reported. For GE inverters the charge rate reports the max
AC rate and thus needs to be reduced by inverter losses.
You can try computing your discharge curve and check recommendations for changing this figure in the logfile.

**input_number.predbat_battery_rate_max_scaling_discharge** is a percentage factor to adjust your maximum discharge rate from that reported by the inverter.
For GE inverters the discharge rate is reported as the max AC rate and thus is fairly accurate.
You can try computing your discharge curve and check recommendations for changing this figure in the logfile.

**switch.predbat_battery_capacity_nominal** - When enabled Predbat uses the reported battery size from the GivTCP 'Battery Nominal Capacity' field
rather than from the normal GivTCP reported 'Battery Capacity kWh' size.
If your battery size is reported wrongly maybe try turning this on and see if it helps.

**input_number.predbat_load_scaling** is a percentage Scaling factor applied to the historical load, increase this if you want to be more pessimistic on future consumption.
Use 1.0 to use exactly the previous load data. A value of 1.1 for example would add 10% to the historical load.
Note that the output data entity predbat.load_energy_h0 will be scaled accordingly.

**input_number.predbat_load_scaling10** is a percentage Scaling factor applied to historical load only for the PV10% scenario (this is in addition to load_scaling above).
This can  be used to make the PV10% scenario take into account extra load usage and hence be more pessimistic while leaving the central scenario unchanged.
The default is 1.1 meaning an extra 10% load is added. This will only have an impact if the PV 10% weighting is non-zero.

**input_number.predbat_load_scaling_saving** is a percentage Scaling factor applied to historical load only during Octopus Saving sessions.
This can be used to model your household cutting down on energy use inside a saving session (e.g. turning off a heat pump, deferring cooking until after the session, etc).

## Solar PV adjustment options

See also [PV configuration options in apps.yaml](apps-yaml.md#solcast-solar-forecast) including an explanation of PV10, PV50 and PV90 terminology.

**input_number.predbat_pv_scaling** is a percentage scaling factor applied to PV data, decrease this if you want to be more pessimistic on PV production vs Solcast.<BR>
Use 1.0 to accurately apply the Solcast forecast generation data. A value of 0.9, for instance, would reduce 10% from the Solcast generation forecast.

**input_number.predbat_pv_metric10_weight** is the percentage weighting given to the Solcast 10% PV scenario in calculating solar generation.
Use 0.0 to disable using the PV 10% in Predbat's forecast of solar generation.
A value of 0.1 assumes that 1 in every 10 times we will get the Solcast 10% scenario, and 9 in every 10 times we will get the 'median' Solcast forecast.<BR>
Predbat estimates solar generation for each half-hour slot to be a pv_metric10_weight weighting of the Solcast 10% PV forecast to the Solcast Median forecast.<BR>
A value of 0.15 is recommended.

## Historical load data

The historical load data is taken from the load sensor as configured in `apps.yaml` and the days are selected
using **days_previous** and weighted using **days_previous_weight** in the same configuration file

**switch.predbat_load_filter_modal** (_expert mode_) when enabled will automatically discard the lowest daily consumption
day from the list of days to use (provided you have more than 1 day selected in days_previous). This can be used to ignore
a single low usage day in your average calculation. By default is feature is enabled but can be disabled only in expert mode.

## Car Charging

There are a number of configuration items in Home Assistant for Predbat to control your car charging.

These are described in detail in [Car Charging](car-charging.md) and are listed here just for completeness:

- **switch.predbat_car_charging_hold** - remove the historical car charging load from the house load
- **input_number.predbat_car_charging_threshold** - power threshold above which Predbat assumes the car is being charged
- **input_number.predbat_car_charging_energy_scale** - used to scale the **car_charging_energy** sensor in apps.yaml
- **input_number.predbat_car_charging_rate** - car's charging rate in kW per hour
- **input_number.predbat_car_charging_loss** - percentage energy lost when charging the car
- **switch.predbat_octopus_intelligent_charging** - controls whether Octopus Intelligent (via the Octopus Energy integration) controls the car charging or Predbat plans the car charging
- **switch.predbat_octopus_intelligent_ignore_unplugged** (_expert mode_) - used with Octopus Intelligent to prevent Predbat from assuming the car will be charging when the car is unplugged
- **binary_sensor.predbat_car_charging_slot** - set to True by Predbat when the car should be charged (Predbat-led charging)
- **select.predbat_car_charging_plan_time** - the time you want the car to be charged by
- **switch.predbat_car_charging_plan_smart** - allows Predbat to allocate car charging slots to the cheapest times rather than all low-rate slots
- **input_number.predbat_car_charging_plan_max_price** - maximum price per kWh to pay when charging your car
- **switch.predbat_car_charging_from_battery** - prevent the car from draining the home battery when charging
- **switch.predbat_car_charging_manual_soc** - ignore the  **car_charging_soc** car SoC sensor set in apps.yaml
- **input_number.predbat_car_charging_manual_soc_kwh** - manually maintained car SoC (used if car SoC sensor is not available)

## Calculation options

See the Predbat mode setting above for basic calculation options

**input_number.predbat_forecast_plan_hours** is the minimum length of the Predbat charge plan, and is the number of hours _after_ the first charge slot to include in the plan.
The default of 24 hours is the recommended value (to match energy rate cycles). Note that the actual length of the Predbat plan will vary depending upon when the first charge slot is.

**switch.predbat_calculate_export_oncharge** (_expert mode_) When True calculated export slots will
disable or move charge slots, allowing them to intermix. When False export slots will never be placed into charge slots.

**switch.predbat_set_discharge_during_charge** - If turned off disables inverter discharge during charge slots, useful for multi-inverter setups
to avoid cross-charging when batteries are out of balance.

**switch.predbat_inverter_set_charge_before** - (_expert_mode_) When True charge slots will be programmed before their start time, when False they will only be
configured when the charging time starts.

**switch.predbat_calculate_tweak_plan** (_expert mode_) When True causes Predbat to perform a second pass optimisation
across the next 8 charge and export windows in time order.

This can help to slightly improve the plan for tariffs like Agile but can make it worse in some fixed rate tariffs which
you want to force export late.

**switch.predbat_calculate_second_pass** (_expert mode_) When True causes Predbat to perform a second pass optimisation
across all the charge and export windows in time order.

NOTE: This feature is quite slow so may need a higher-performance machine.

This can help to slightly improve the plan for tariffs like Agile but can make it worse in some fixed rate tariffs in which you want to force export late.

**switch.calculate_import_low_export** (_expert_mode_) When True import slots of the same value are sorted by export price.
When False they are sorted just by price and then time. The default is True.

By default with this option enabled if there are multiple charge slots of the same price Predbat will try to charge when the export rates are the lowest
thus leaving the higher export slots available.

**switch.calculate_export_high_import** (_expert_mode_) When True export slots of the same value are sorted by import price (to avoid the low import slots for export).
When False they are sorted just by price and then time. The default is True.

By default with this option disabled the latest export slot of the same value will be picked, this is useful for fixed-price export tariffs where
you want to export as late in the day as you can.

## Battery margins and metrics options

**input_number.predbat_best_soc_keep** is the minimum battery level in kWh that Predbat will to try to keep the battery above for the Predbat plan.
This is a soft constraint only that's used for longer-term planning and is ignored for the forthcoming first 4 hours of the plan.
As this is not used for short-term planning your SoC can drop below this - use **input_number.predbat_best_soc_min**
if you want to force all charges to be above a set level.
It's usually good to have best_soc_keep set to 0 to use your entire battery, values above zero will end up costing more money but will ensure you
have some spare battery kept back for emergency use.

**input_number.predbat_best_soc_keep_weight** (_expert_mode_) Is used to tune how strongly you want the keep metric to apply.
A value of 0 would essentially ignore keep while higher values will make it more important to always stay above your keep threshold even if it costs
more money to do so.

The default is 0.5 - this is the recommended setting.

**input_number.predbat_best_soc_min** (_expert mode_) sets the minimum charge level (in kWh) for charging during each slot and the
minimum force export level also (set to 0 if you want to skip some slots).
If you set this to a non-zero value you will need to use the low rate threshold to control which slots you charge from or you may charge all the time.

**input_number.predbat_best_soc_max** (_expert mode_) sets the maximum charge level (in kWh) for charging during each slot.
A value of 0 disables this feature.

**input_number.combine_rate_threshold** (_expert mode_) sets a threshold (in pence) to combine charge or export slots into a single larger average rate slot.
The default is 0p which disables this feature and all rate changes result in a new slot.

**switch.predbat_combine_charge_slots** Controls if charge slots of > 30 minutes can be combined. When disabled they will be split up,
increasing run times but potentially more accurate for planning. Turn this off if you want to enable ad-hoc import
during long periods of higher rates but you wouldn't charge normally in that period (e.g. pre-charge at the day rate before
a saving session). The default is disabled (False)

**switch.predbat_combine_export_slots** (_expert mode_) Controls if export slots of > 30 minute can be combined. When disabled
they will be split up, increasing run times but potentially more accurate for planning. The default is disabled (False)

**input_number.predbat_metric_min_improvement** (_expert mode_) sets the minimum cost improvement in pence that it's worth lowering the battery SOC % for.
The default value is 0 which means this feature is disabled and the battery will be charged less if it's cost neutral.
If you use **input_number.predbat_pv_metric10_weight** then you probably don't need to enable this as the 10% forecast does the same thing better
Do not use it if you have multiple charge windows in a given period as it won't lead to good results (e.g. Agile)
You could even go to something like -0.1 to say you would charge less even if it cost up to 0.1p more (best used with metric10).

**input_number.predbat_metric_min_improvement_export** (_expert mode_) Sets the minimum pence cost improvement it's worth doing a forced export for.
A value of 0.1 is the default which prevents any marginal exports as they must be worth at least 0.1 pence for a 30-minute slot (less for shorter slots).
If you increase this value (e.g. you only want to force export if very profitable), then exports will become less common.
The value is in pence per 30 minutes of export time.

**input_number.predbat_metric_min_improvement_export_freeze** (_expert mode_) Sets the minimum pence cost improvement it's worth doing an export freeze for.
A value of 0.1 is the default which prevents any marginal freezes as they must be worth at least 0.1 pence for a 30-minute slot (less for shorter slots).
If you increase this value (e.g. you only want to freeze export if very profitable), then freeze exports will become less common.
The value is in pence per 30 minutes of export time.

**input_number.predbat_metric_min_improvement_swap** (_expert mode_) Sets the minimum improvement in cost to swap an export slot to a later time in the day.
The default for this setting is -0.25p, meaning that export will be moved later even if it costs up to an extra 0.25p on the plan.
The advantage of having late exports means that you are less likely to run out of battery earlier in the day.
Values of between -1p and 0p would be recommended.

**input_number.predbat_rate_low_threshold** (_expert mode_) When set to 0 (the default) Predbat will automatically look at the future import rates in the plan
and determine the import rate threshold below which a slot will be considered to be a potential charging slot.<BR>
If rate_low_threshold is set to a non-zero value this will set the threshold below future average import rates as the minimum to consider for a charge window,
e.g. setting to 0.8 = 80% of the average rate.<BR>
If you set this too low you might not get enough charge slots. If it's too high you might get too many in the
24-hour period which makes optimisation harder.

**input_number.predbat_rate_high_threshold** (_expert mode_) When set to 0 (the default) Predbat will automatically look at the future export rates in the plan
and determine the threshold above which a slot can be considered a potential exporting slot.<BR>
If rate_high_threshold is set to a non-zero value this will set the threshold above future average export rates as the minimum export rate to consider exporting for,
e.g. setting to 1.2 = 20% above average rate.<BR>
If you set this too high you might not get any export slots. If it's too low you might get too many in the 24-hour period.

**input_number.predbat_metric_future_rate_offset_import** (_expert mode_) Sets an offset to apply to future import energy rates that are
not yet published, best used for variable rate tariffs such as Agile import where the rates are not published until 4pm.
If you set this to a positive value then Predbat will assume unpublished import rates are higher by the given amount.

Setting this to 1 to 1.5p for example results in Predbat being a little more aggressive in the charging calculation for today -
Predbat will charge the battery to a higher percentage than it would otherwise as it expects a cost-benefit of using today's lower rates.
NB: this can lead to higher costs and to some export if solar generation is better than forecast.

**input_number.predbat_metric_future_rate_offset_export** (_expert mode_) Sets an offset to apply to future export energy rates that are
not yet published, best used for variable rate tariffs such as Agile export where the rates are not published until 4pm.
If you set this to a negative value then Predbat will assume unpublished export rates are lower by the given amount.

**switch.predbat_calculate_inday_adjustment** (_expert mode_) Enabled by default with damping of 0.95. When enabled will
calculate the difference between today's actual load and today's predicated load and adjust the rest of the day's usage
prediction accordingly. A scale factor can be set with **input_number.predbat_metric_inday_adjust_damping** (_expert mode_)
to either scale up or down the impact of the in-day adjustment (lower numbers scale down its impact). The in-day adjustment
factor can be seen in **predbat.load_inday_adjustment** and charted with the In-Day Adjustment chart (template can be found
in the charts template in Github).

**switch.predbat_metric_pv_calibration_enable** When enabled allows Predbat to use historical data to calibrate your PV
production estimates on a 30 minute basis based on actual data. This can be useful to adjust for your systems real performance.
Default is Off.

**input_number.predbat_carbon_metric** (_carbon enable_) When Carbon footprint tracking is enabled (**switch.predbat_carbon_enable**)
you can specify a cost per Kg of CO2 used to weight the selection of plans. Values of around 10-200 will give varying outcomes to trade off
cost vs carbon footprint of your system.

_Note: Carbon footprint tracking can only be enabled if apps.yaml is configured to point to the correct CO2 cost sensor_

## Inverter control options

**switch.predbat_set_status_notify** Enables mobile notification about changes to the Predbat state (e.g. Charge, Export etc). On by default.

**switch.predbat_set_inverter_notify** Enables mobile notification about all changes to inverter registers (e.g. setting window, turning discharge on/off).
Off by default.

**switch.predbat_set_charge_low_power** Enables low-power charging mode where the max charge rate will be automatically determined by Predbat  to be the
lowest possible rate to meet the charge target. This is only really effective for charge windows longer than 30 minutes.
If this setting is turned on, it is strongly recommended that you create a [battery_power_charge_curve in apps.yaml](apps-yaml.md#workarounds)
as otherwise the low power charge may not reach the charge target in time.
This setting is off by default.

The YouTube video [low power charging and charging curve](https://youtu.be/L2vY_Vj6pQg?si=0ZiIVrDLHkeDCx7h)
explains how the low-power charging works and shows how Predbat automatically creates it.

**input_number.predbat_charge_low_power_margin** (requires switch.predbat_set_charge_low_power to be enabled) Controls how many minutes before the completion time to target finishing charging, this defaults to 10
but can be changed between 0 and 30.

**switch.predbat_set_reserve_enable** (_expert_mode_) When enabled the reserve setting is used to hold the battery charge level
once it has been reached or to protect against discharging beyond the set limit. Enabled by default.

**switch.predbat_set_export_freeze** When enabled will allow Predbat to export Solar to the grid rather than charging the battery.
Enabled by default on those inverters that have this support.

**switch.predbat_set_charge_freeze** (_expert mode_) When enabled will allow Predbat to hold the current battery level while drawing
from the grid/solar as an alternative to charging. Enabled by default.

**switch.predbat_set_export_freeze_only** (_expert mode_) When enabled forced export is prevented, but export freeze can be used
(if enabled) to export excess solar rather than charging the battery. This is useful with tariffs that pay you for
solar exports but don't allow forced export (brown energy).

If you have **switch.predbat_inverter_hybrid** set to False then if **switch.predbat_inverter_soc_reset** (_expert mode_) is set to True then the
target SOC % will be reset to 100% outside of a charge window. This may be required for the AIO inverter to ensure it charges from solar. The default for
this switch is True but it can be disabled in expert mode if need be.

**input_number.predbat_set_reserve_min** Defines the battery reserve percentage, i.e. the minimum charge level that should be retained in the battery,
a value of 4 is the minimum and recommended to make use of the full battery.
When the battery level reaches the specified minimum level, your inverter will start importing from the grid to satisfy house load.
If you want to pre-prepare the battery to retain extra charge in the event of a high likelihood of a grid power outage such as storms predicted,
you can increase set_reserve_min to a higher value, and then change it back afterwards.<BR>
(Obviously this is only any use if your inverter is wired to act as an Emergency Power Supply or whole-home backup 'island mode' on the GivEnergy AIO).

Note that input_number.predbat_set_reserve_min is used by Predbat to stop the battery SoC dropping below the specified level but it is not used in planning battery charging.
i.e. if you change the reserve min to a level higher than the current battery SoC then Predbat will not plan charging activity to increase the SoC to that level; you should set [manual charges](#manual-control) to increase the SoC level.

**switch.predbat_inverter_soc_reset**  (_expert mode_) When enabled the target SOC for the inverter(s) will be reset to 100%
when a charge slot is not active, this can be used to workaround some firmware issues where the SOC target is
used for solar charging as well as grid charging. When disabled the SOC % will not be changed after a charge slot.
This is enabled by default.

## Balance Inverters

When you have two or more inverters it's possible they get out of sync so they are at different charge levels or they start to cross-charge (one discharges into another).
When enabled, balance inverters try to recover this situation by disabling either charging or discharging from one of the batteries until they re-align.

The `apps.yaml` contains a setting **balance_inverters_seconds** which defines how often to run the balancing, 30 seconds is recommended if your
machine is fast enough, but the default is 60 seconds.

Enable the **switch.predbat_balance_inverters_enable** switch in Home Assistant to enable this feature.

- **switch.predbat_balance_inverters_charge** - Is used to toggle on/off balancing while the batteries are charging
- **switch.predbat_balance_inverters_discharge** - Is used to toggle on/off balancing while the batteries are discharging
- **switch.predbat_balance_inverters_crosscharge** - Is used to toggle on/off balancing when the batteries are cross charging
- **input_number.predbat_balance_inverters_threshold_charge** - Sets the minimum percentage divergence of SoC during charge before balancing, default is 1%
- **input_number.predbat_balance_inverters_threshold_discharge** - Sets the minimum percentage divergence of SoC during discharge before balancing, default is 1%

## Cloud coverage and load variance

Predbat tries to model passing clouds by modulating the PV forecast data on a 5-minute interval up and down while retaining the same predicted total.
The amount of modulation depends on the difference between the PV50% (default) and PV10% scenario produced by Solcast.

You can disable this feature (_expert mode only_) using **switch.predbat_metric_cloud_enable**

Predbat tries to model changes in your household load by modulating the historical data on a 5-minute interval up and down while retaining the same predicted total.
The amount of modulation depends on the standard deviation of your load predictions over the coming period (currently 4 hours).

You can disable this feature (_expert mode only_) using **switch.metric_load_divergence_enable**

## iBoost model options

Predbat has an 'iBoost model' that can be used to model using excess solar energy to heat hot water (or similar) instead of it being exported to the grid.

This model can be used to control any solar diverter device, for example, an iBoost (e.g. using a Fingerbot or similar device to physically press the 'boost' button on the iBoost),
a MyEnergy Eddi (using the MyEnergy integration), or it can be used with a high-power smart switch to turn on the hot water cylinder immersion heater when there is excess solar.

So although Predbat refers to controlling an iBoost, you are not limited to just an iBoost device when using this model within Predbat.

To turn the model on, **switch.predbat_iboost_enable** needs to be enabled.

The predicted output from the iBoost solar diverter model is returned in **predbat.iboost_best** and is populated in the 'iBoost' column of the [Predbat plan](predbat-plan-card.md).

### iBoost basic configuration

When you turn on predbat_iBoost_enable the following additional Home Assistant entities are created by Predbat:

- **input_number.predbat_iboost_max_energy** Sets the maximum energy in kWh that the solar diverter can consume during a day before turning off - default 3kWh.

- **input_number.predbat_iboost_max_power** Sets the maximum power in watts that the solar diverter will consume - default 2400.

- **input_number.predbat_iboost_min_power** Sets the minimum power in watts that the solar diverter will consume - default 500.

- **input_number.predbat_iboost_value_scaling** Sets how to account for the value of iBoost units of energy.
The default value of 0.75 means that each kWh of energy diverted is accounted for as 0.75 times The lowest future import rate.

Higher values will generate plans with more solar diversion while lower values will generate less.
A value of 0 means all diverted energy should be ignored in planning (assumed to be zero value).

- **switch.predbat_iboost_energy_subtract** When enabled (default) energy reported by the **iboost_energy_today** sensor configured in apps.yaml will be subtracted
from your historical load data to improve the accuracy of predictions.
If you are using PredAI you will also need to subtract this data inside PredAI configuration.

Different boost modes can be selected:

### iBoost modes

#### Solar

- **switch.predbat_iboost_solar** When enabled assume the diverter will use solar power to boost the hot water heating.
- **switch.predbat_iboost_solar_excess** When disabled will allow any solar power to be diverted while if enabled only excess power (power that would otherwise be exported) will be used. The default is Off.
- By default iboost will not run the battery is in force export mode unless you turn on **switch.predbat_iboost_on_export**

- **input_number.predbat_iboost_min_soc** sets the minimum home battery SoC percentage that must be in the battery before the solar diverter is turned on.
The default is 0 meaning hot water heating can occur regardless of what SoC level the battery is at.

#### Battery

- **switch.predbat_iboost_charging** If set to on, the diverter will operate when the battery is charging (can be combined with solar)

#### Energy rates only

If both of the above are off, but iBoost is enabled then boost will happen solely based on energy rates (see below).

- **switch.predbat_iboost_smart** Will pick the lowest import rate slots within 24 hours to achieve the desired energy as specified by **input_number.predbat_iboost_max_energy**.

Only slots at or below the rate threshold will be selected.

Note this option only applies when iboost_solar and iboost battery are both off.

- **input_number.predbat_iboost_smart_min_length** Sets the minimum slot length in minutes to iBoost (only applies for energy rate only modes).
The default is 30 minutes but can be set in multiples of 30. Increasing this slot size could increase costs depending on your tariff.

- **switch.predbat_iboost_on_export** If set to on allows iBoost to run even if the battery is forced to export to the grid, otherwise it won't run in these circumstances.

- **switch.iboost_prevent_discharge** When set will stop your battery from discharging when iBoost is active and thus prevent your battery from draining to the diverter.
This switch will work in all modes and is not recommended to be used when iBoost Solar is enabled as it will prevent your battery from discharging during excess solar periods
which could cause additional imports due to passing clouds.

### iBoost energy rate filtering

- **input_number.predbat_iboost_rate_threshold** Sets the maximum import rate (in pence) that the diverter will trigger on, defaults to 100.
- **input_number.predbat_iboost_rate_threshold_export** Sets the maximum export rate (in pence) that the diverter will trigger on, defaults to 100.

- **switch.predbat_iboost_gas** When enabled will control the diverter to only operate when import electric rates are lower than gas rates.
- **switch.predbat_iboost_gas_export** When enabled will control the diverter to only operate when export electric rates are lower than gas rates.

These can be useful if you have the choice to heat your hot water by immersion heater or by gas boiler.

Note: Gas rates have to be configured in `apps.yaml` using **metric_octopus_gas** or **rates_gas** depending upon whether you are using the Octopus Integration or not.

- **input_number.predbat_iboost_gas_scale** Sets the scaling of the gas rates (set in `apps.yaml`)used before comparing with electric rates, to account for gas boiler losses and efficiency.

It should be set to the reciprocal of the boiler efficiency, i.e. for an 80% efficient gas boiler, set to 1.25.

### iBoost control

### iBoost output data

You will see **input_number.predbat_iboost_today** entity which tracks the estimated kWh consumed by the solar diverter during the day, and resets at midnight every night.

The **binary_sensor.predbat_iboost_active** entity will be enabled when the solar diverter should be active and can be used for automations to trigger the immersion heater boost.

The attributes within this sensor include 'solar' which includes Solar diversion should be active and 'full' which indicates iBoost should run at maximum rate (could be
during a charge cycle or grid import).

Example template automation for controlling the solar diverter:

```yaml
alias: Solar Diverter
description: "Start/stop solar diverter based on Predbat-determined slots"
trigger:
  - platform: state
    entity_id:
     - binary_sensor.predbat_iboost_active
action:
  - choose:
      - conditions:
          - condition: state
            entity_id: binary_sensor.predbat_iboost_active
            state: "True"
        sequence:
          <commands to turn on your solar diverter>
      - conditions:
          - condition: state
            entity_id: binary_sensor.predbat_iboost_active
            state: "False"
        sequence:
          <commands to turn off your solar diverter>
mode: single
```

If you have an incrementing sensor that tracks the solar diverter energy usage then to make your predictions more accurate you should set the **iboost_energy_today** sensor in
`apps.yaml` to point to it, and optionally set **iboost_energy_scaling** if the sensor isn't in kWh (e.g. set to 0.001 if the sensor is in Watts).
The sensor should be an incrementing sensor which can reset at midnight or not.

## Holiday mode

When you go away you are likely to use less electricity and so the previous load data will be quite pessimistic.

Using the Home Assistant entity **input_number.predbat_holiday_days_left** you can set the number of full days that
you will be away (including today). The number will count down by 1 day at midnight until it gets back to zero.
Whilst the holiday days left are non-zero, Predbat's 'holiday mode' is active.

When Predbat's 'holiday mode' is active the historical load data will be taken from yesterday's data (1 day ago) rather than from the **days_previous** setting in `apps.yaml`.
This means Predbat will adjust more quickly to the new usage pattern.

If you have been away for a longer period (more than your normal days_previous setting) then obviously it's going
to take longer for the historical data to catch up, you could then enable holiday mode for another 7 days after your return.

In summary:

- For short holidays set holiday_days_left to the number of full days you are away, including today but excluding the return day
- For longer holidays set holiday_days_left to the number of days you are away plus another 7 days until the data catches back up

## Manual control

In some cases, you may want to override Predbat's planned behaviour and make a decision yourself. One way to achieve this is to put Predbat into
read-only mode using **switch.predbat_set_read_only**. When going to read-only mode the inverter will be put back to the default settings and you should then
control it yourself using GivTCP or the App appropriate to your inverter.

A better alternative in some cases is to tell Predbat what you want it to do using the manual force features:

You can force the battery to be charged within a 30-minute slot by using the **select.predbat_manual_charge** selector.
Pick the 30-minute slot you wish to charge in, and Predbat will change the plan to charge in the selected slot.
You can select multiple slots by using the drop-down menu more than once.
When Predbat updates the plan you will see the slots picked to be charging slots in the current value of this selector,
and annotated in the [Predbat HTML plan](predbat-plan-card.md#displaying-the-predbat-plan) with an upside down 'F' symbol.

You can cancel a force slot by selecting the slot time again (it will be shown in square brackets to indicate it's already selected).

![image](https://github.com/springfall2008/batpred/assets/48591903/aa668cc3-60fc-4956-8619-822f09f601dd)

The **select.predbat_manual_export** selector can be used to manually force an export within a 30-minute slot in the same way as the manual force charge feature.
The force export takes priority over force charging.

The **select.predbat_manual_demand** selector is used to force Predbat to demand mode during a 30-minute slot, this implies no forced grid charging or exporting of the battery.
House load will be supplied from solar, or the battery if there is insufficient solar, or grid import if there is insufficient battery charge.
This is described as 'ECO' Mode for GivEnergy inverters but other inverters use different terminology.

The **select.predbat_manual_freeze_charge** selector is used to force Predbat to freeze charge during a 30-minute slot, this implies the battery will not discharge and
hold at the current level. The grid may be used if solar is not enough to cover the load.

The **select.predbat_manual_freeze_export** selector is used to force Predbat to freeze export during a 30-minute slot, this implies the battery will not charge but will
still discharge for the house load. Any solar will be exported to the grid.

When you use the manual override features you can only select times in the next 18 hours, the overrides will be removed once their time
slot expires (they do not repeat).

_NOTE: once you select a time slot from any of the **select.predbat_manual_** selectors the selected time slot is immediately marked on the drop-down and you can then make another change.
Predbat still has to update the plan which it will be doing so in the background,
and this can take a few minutes to run (depending on the speed and power of the PC you are running Home Assistant on) so don't be surprised why the
[Predbat plan](predbat-plan-card.md) doesn't change immediately - remember you can see the date/time the plan was last updated on the first row of the plan.

_CAUTION: If you leave Predbat turned off for a long period of time then the override timeslots could end up repeating when you restart_

![image](https://github.com/springfall2008/batpred/assets/48591903/7e69730f-a379-483a-8281-f72de0cc6e97)

## Debug

**switch.predbat_debug_enable** When on will create lots of debugging information to aid diagnosis of Predbat issues.  By default this is turned off and its recommended that its only switched on when debug logs are requested. With the switch on:

- Firstly Predbat prints lots of extra debug information in the predbat logfile which means the logfile will fill and swap to a new file more rapidly
- Secondly Predbat will create a debug output file 'debug/predbat_debug_HH_MM_SS.yaml' in a subfolder of the Predbat installation directory.
This file contains a full export of your current Predbat config and is extremely useful to enable recreating your setup to diagnose issues. Any sensitive information such as Solcast or GivEnergy Cloud API keys are automatically removed.

The following automation might be useful to automatically turn off Predbat debug mode after turning it on to capture the debug logs:

```yaml
alias: "Predbat: Auto turn-off debug mode"
description: ""
triggers:
  - trigger: state
    entity_id:
      - switch.predbat_debug_enable
    to: "on"
    for:
      minutes: 10
conditions: []
actions:
  - action: script.notify_all_devices
    metadata: {}
    data:
      title: "INFO: Turned off Predbat debug after 10 minutes"
      critical: "N"
      url: ""
      message: switch.predbat_debug_enable turned off
  - action: switch.turn_off
    target:
      entity_id: switch.predbat_debug_enable
    data: {}
mode: single
```

**switch.predbat_plan_debug** (_expert mode_) when enabled adds some extra debug to the Predbat HTML plan - see [Predbat Plan debug mode](predbat-plan-card.md#debug-mode-for-predbat-plan)
for more details.
