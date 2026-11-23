#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Predbat Home Battery System
# Copyright Trefor Southwell 2026 - All rights reserved
# This application maybe used for personal use only and not for commercial use
# -----------------------------------------------------------------------------
# fmt: off
"""
Plan accuracy check: compare what a plan predicted with what then happened.

For a folder of Predbat debug dumps (for example the hourly ones), this replays each dump with
a setting on and off, reads the plan's predicted battery SoC for the time of each later dump,
and compares it with the real SoC recorded in that later dump.

It is an evidence-gathering tool, it is not part of Predbat itself.

Run it from the coverage directory of a Predbat checkout (the replay needs the test setup there):

    cd coverage
    python3 ../tools/plan_accuracy.py /path/to/dumps
    python3 ../tools/plan_accuracy.py /path/to/dumps --option fleet_power_curves_enable --max-horizon 8

Replays take around ten seconds each and are cached in plan_accuracy_cache/ next to the dumps, so
adding new dumps and running again only replays the new ones.

It also prints a second view of the plans that were really live, split by whether the option was on or off at the
time (read from each dump), for an on-days versus off-days test. That view never uses a plan that was not run.

Notes:
- A replay re-plans from the dump's own inputs, it is not what the live system had planned
  at that moment (although it should be very close).
- The error is predicted minus actual fleet SoC in percentage points of the whole fleet, so
  positive means the plan expected more energy in the batteries than there was.
- Settings such as reserves or overrides that you change between two dumps are real events
  that the earlier plan could not know about, so they show up as error.
"""
# fmt: on

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import statistics
import sys
from datetime import datetime, timedelta

INVERTER_FIELDS = ("id", "soc_kw", "soc_max", "reserve")


def read_dump_state(path, option=None):
    """
    Read the time and the per-inverter battery state out of a debug dump without loading all of it.

    Returns a dict with 'time' (datetime), 'soc_kw' and 'soc_max' (fleet totals in kWh), 'inverters'
    (a list of dicts with id, soc_kw, soc_max and reserve) and 'live' (the value the named boolean option
    had when the dump was taken, True or False, or None if the dump does not record it), or None if the
    dump has no usable state.
    """
    now_utc = None
    live = None
    option_line = re.compile(r"^{}:\s*(true|false)\s*$".format(re.escape(option)), re.IGNORECASE) if option else None
    inverters = []
    in_inverters = False
    current = None
    with open(path, "r", errors="replace") as handle:
        for line in handle:
            if option_line:
                found = option_line.match(line)
                if found:
                    live = found.group(1).lower() == "true"
                    continue
            if not in_inverters:
                if line.startswith("now_utc:"):
                    now_utc = line.split(":", 1)[1].strip()
                elif line.startswith("inverters:"):
                    in_inverters = True
                continue
            if line.startswith("- "):
                current = {}
                inverters.append(current)
                line = "  " + line[2:]
            elif not line.startswith("  "):
                in_inverters = False
                continue
            match = re.match(r"^  (id|soc_kw|soc_max|reserve):\s*(\S+)", line)
            if match and current is not None:
                current[match.group(1)] = float(match.group(2))
    if not now_utc or not inverters:
        return None
    when = datetime.fromisoformat(now_utc.replace("Z", "+00:00"))
    inverters = [inv for inv in inverters if "soc_kw" in inv and "soc_max" in inv]
    if not inverters:
        return None
    return {
        "time": when,
        "soc_kw": sum(inv["soc_kw"] for inv in inverters),
        "soc_max": sum(inv["soc_max"] for inv in inverters),
        "inverters": inverters,
        "live": live,
    }


def replay(dump, option, value, cache_dir, coverage_dir):
    """
    Replay a dump with option=value and return the plan rows, using a cached copy where there is one.
    """
    cache_file = os.path.join(cache_dir, "{}.{}={}.json".format(os.path.basename(dump), option, value))
    if os.path.exists(cache_file):
        with open(cache_file) as handle:
            return json.load(handle)
    unit_test = os.path.join(coverage_dir, "..", "apps", "predbat", "unit_test.py")
    command = [sys.executable, unit_test, "--debug_file", os.path.abspath(dump), "--override", "{}={}".format(option, value)]
    result = subprocess.run(command, cwd=coverage_dir, capture_output=True, text=True)
    plan_file = os.path.join(coverage_dir, "plan_final.json")
    if result.returncode != 0 or not os.path.exists(plan_file):
        print("Replay failed for {} ({}={}): {}".format(dump, option, value, (result.stdout + result.stderr)[-300:]))
        return None
    # Keep only what is needed, plan_final.json is overwritten by the next replay
    with open(plan_file) as handle:
        plan = json.load(handle)
    rows = [{"time": row["time"], "soc_percent": row["soc_percent"], "state": row.get("state")} for row in plan["rows"]]
    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_file, "w") as handle:
        json.dump(rows, handle)
    return rows


def predicted_percent(rows, when, tolerance_minutes=15):
    """
    The plan's predicted fleet SoC percent for the row nearest to 'when', or None if no row is within tolerance.
    """
    best = None
    best_gap = None
    for row in rows:
        row_time = datetime.strptime(row["time"], "%Y-%m-%dT%H:%M:%S%z")
        gap = abs((row_time - when).total_seconds()) / 60.0
        if best_gap is None or gap < best_gap:
            best, best_gap = row, gap
    if best is None or best_gap > tolerance_minutes:
        return None
    return float(best["soc_percent"])


def print_live_view(live_errors, option):
    """
    Print the error of the plans that were really live, split by which setting was live, by horizon band and by day.

    This is the view for an on-days versus off-days test: it never uses a plan that was not run, so it is not
    confounded by what the batteries did following the other setting.
    """
    if not live_errors:
        print("\nNo dump records {} as live, so there is no live-setting view".format(option))
        return
    bands = ((0, 2), (2.5, 4), (4.5, 6), (6.5, 8))
    print("\nPlans that were live, by the setting live at the time ({} - predicted minus actual, percentage points)".format(option))
    print("{:>12} {:>8}  {:>4} {:>8} {:>8} {:>10}".format("setting", "horizon", "n", "mean", "abs", "median abs"))
    for live in (True, False):
        for low, high in bands:
            errors = [err for _, horizon, setting, err in live_errors if setting == live and low <= horizon <= high]
            if not errors:
                continue
            print(
                "{:>12} {:>3.1f}-{:<3.1f}h  {:>4} {:>+8.1f} {:>8.1f} {:>10.1f}".format(
                    "on" if live else "off", low, high, len(errors), sum(errors) / len(errors), sum(abs(x) for x in errors) / len(errors), statistics.median(abs(x) for x in errors)
                )
            )
    print("\nBy day the plan was made (all horizons up to the maximum)")
    print("{:>8} {:>8}  {:>4} {:>8} {:>8} {:>10}".format("day", "setting", "n", "mean", "abs", "median abs"))
    days = sorted({(start.date(), setting) for start, _, setting, _ in live_errors})
    for day, setting in days:
        errors = [err for start, _, live, err in live_errors if start.date() == day and live == setting]
        print("{:>8} {:>8}  {:>4} {:>+8.1f} {:>8.1f} {:>10.1f}".format(day.strftime("%d %b"), "on" if setting else "off", len(errors), sum(errors) / len(errors), sum(abs(x) for x in errors) / len(errors), statistics.median(abs(x) for x in errors)))


def main():
    """
    Replay the dumps and print the error table.
    """
    parser = argparse.ArgumentParser(description="Compare plan predicted SoC with later actual SoC from Predbat debug dumps")
    parser.add_argument("dumps", help="Folder of debug dumps (predbat_debug_*.yaml or *.yaml.txt)")
    parser.add_argument("--option", default="fleet_power_curves_enable", help="Boolean setting to compare on and off")
    parser.add_argument("--max-horizon", type=float, default=8.0, help="Longest look-ahead to compare, in hours")
    args = parser.parse_args()

    coverage_dir = os.getcwd()
    if not os.path.exists(os.path.join(coverage_dir, "apps.yaml")):
        print("Run this from the coverage directory of a Predbat checkout")
        return 1
    files = sorted(glob.glob(os.path.join(args.dumps, "*.yaml")) + glob.glob(os.path.join(args.dumps, "*.yaml.txt")))
    states = []
    for path in files:
        state = read_dump_state(path, args.option)
        if state:
            states.append((state["time"], path, state))
        else:
            print("Skipping {}: no inverter state found".format(path))
    states.sort(key=lambda item: item[0])
    if len(states) < 2:
        print("Need at least two dumps")
        return 1
    cache_dir = os.path.join(args.dumps, "plan_accuracy_cache")

    results = {True: {}, False: {}}
    detail = []
    live_errors = []
    for index, (start, path, state) in enumerate(states):
        later = [item for item in states[index + 1 :] if item[0] - start <= timedelta(hours=args.max_horizon)]
        if not later:
            continue
        for value in (True, False):
            rows = replay(path, args.option, value, cache_dir, coverage_dir)
            if rows is None:
                continue
            for end, _, end_state in later:
                predicted = predicted_percent(rows, end)
                if predicted is None:
                    continue
                actual = 100.0 * end_state["soc_kw"] / end_state["soc_max"]
                horizon = round((end - start).total_seconds() / 3600.0 * 2) / 2.0
                error = predicted - actual
                results[value].setdefault(horizon, []).append(error)
                detail.append((start, end, value, predicted, actual))
                # The plan that was really live: the replay with the setting the dump had, scored only where the
                # setting did not change between the two dumps, so each comparison belongs to one setting
                if state["live"] is not None and value == state["live"] and end_state["live"] == state["live"]:
                    live_errors.append((start, horizon, state["live"], error))

    print("\nPredicted minus actual fleet SoC, percentage points ({} on versus off)".format(args.option))
    print("mean is the average signed error, abs the average size of the error, median abs the typical size (one odd plan can swing the means)")
    print("{:>8}  {:>4}  {:>8} {:>8} {:>10}   {:>8} {:>8} {:>10}".format("horizon", "n", "on mean", "on abs", "on med abs", "off mean", "off abs", "off med abs"))
    for horizon in sorted(set(results[True]) | set(results[False])):
        on = results[True].get(horizon, [])
        off = results[False].get(horizon, [])
        count = min(len(on), len(off))
        if not count:
            continue
        print(
            "{:>7.1f}h  {:>4}  {:>+8.1f} {:>8.1f} {:>10.1f}   {:>+8.1f} {:>8.1f} {:>10.1f}".format(
                horizon,
                count,
                sum(on) / len(on),
                sum(abs(x) for x in on) / len(on),
                statistics.median(abs(x) for x in on),
                sum(off) / len(off),
                sum(abs(x) for x in off) / len(off),
                statistics.median(abs(x) for x in off),
            )
        )
    print_live_view(live_errors, args.option)

    print("\nEach comparison")
    print("{:>16} {:>16}   {:>9} {:>9} {:>9}".format("plan from", "compared at", "on", "off", "actual"))
    by_pair = {}
    for start, end, value, predicted, actual in detail:
        by_pair.setdefault((start, end), {"actual": actual})[value] = predicted
    for (start, end), item in sorted(by_pair.items()):
        print("{:>16} {:>16}   {:>8.1f}% {:>8.1f}% {:>8.1f}%".format(start.strftime("%d %b %H:%M"), end.strftime("%d %b %H:%M"), item.get(True, float("nan")), item.get(False, float("nan")), item["actual"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
