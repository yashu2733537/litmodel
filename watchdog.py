#!/usr/bin/env python3
"""litmodel training watchdog -- watches 24/7, auto-fixs what's safe.

Rules:
  - training process dead (no halt flag)  -> AUTO-RESTART --resume (max 3/h)
  - NaN/inf loss                          -> kill + halt + ALERT (needs diagnosis)
  - val rising >=6/7 consecutive, net>=.04 (post-iter-40) -> kill + halt + ALERT
  - log silent >30 min (process alive)    -> ALERT; >60 min -> kill + AUTO-RESTART
  - on battery                            -> ALERT (you lose 40% speed)
  - MemAvailable <150MB or disk <5GB      -> ALERT
  - health line every 10 min              -> watchdog.log
  - suspend/resume detected               -> 25-min silence grace; iter-stall and
                                             no-eval timers reset (a long sleep
                                             is not counted against them)
  - iter stalled >90 min (process alive)  -> kill + halt + ALERT (no progress)
  - no parseable evals >30 min            -> ALERT; >60 min -> kill + AUTO-RESTART
  - current run says "training complete." -> DONE (no restart, no error halt)
  - 3+ consecutive tick errors            -> ALERT (detection degraded)

Timing rationale (ISSUE 1): config.py eval_interval=12 prints a new eval line
every ~14-15 min on AC and ~20-22 min on BATTERY (~40% slower). The alert at
30 min and the kill at 60 min both sit above that worst-case normal gap (the
kill is >= 2x it), so healthy battery-speed runs are never false-killed; the
90-min iter-stall budget is ~4x the slowest eval cycle.

Control:  touch ~/litmodel/logs/.halt    = watchdog goes quiet, never restarts
          rm   ~/litmodel/logs/.halt    = re-arm
State:    ~/litmodel/logs/STATUS         = one-line current verdict
          ~/litmodel/logs/ALERT.log      = append-only human-visible alerts
"""
import math
import os
import re
import shutil
import signal
import subprocess
import time
from datetime import datetime

ROOT = os.path.expanduser("~/litmodel")
LOGS = os.path.join(ROOT, "logs")
TRAIN_LOG = os.path.join(LOGS, "train.log")
WD_LOG = os.path.join(LOGS, "watchdog.log")
STATUS = os.path.join(LOGS, "STATUS")
HALT = os.path.join(LOGS, ".halt")
RESTARTS = os.path.join(LOGS, ".restarts")
ALERT_LOG = os.path.join(LOGS, "ALERT.log")  # FIX 8: append-only, never truncated
LAUNCH = ('cd ~/litmodel && echo "=== RUN $(date) ===" | tee -a ~/litmodel/logs/train.log '
          "&& .venv/bin/python -u train.py --preset 10m --resume 2>&1 "
          "| tee -a ~/litmodel/logs/train.log")

# ISSUE 1: cadence-aware silence budgets. config.py raised the 10m preset's
# eval_interval from 4 to 12, so a fresh eval line lands every ~14-15 min on
# AC and ~20-22 min on BATTERY (~40% slower). The old 12/20-min thresholds
# sat BELOW the battery cadence: the ALERT fired constantly in normal
# operation and the 20-min kill could FALSE-KILL a healthy battery run (up to
# 50 iters lost AND it feeds the 6-in-6h restart limiter). Budgets now sit
# above the worst-case normal gap, with the kill >= 2x that gap.
SILENT_ALERT_S = 30 * 60    # alert once silence exceeds one battery eval cycle
SILENT_KILL_S = 60 * 60     # kill at ~2x that -> battery-speed runs survive
STALL_BUDGET_S = 90 * 60    # iter stall: ~4x the slowest (~22 min) eval cycle
SUSPEND_GRACE_S = 25 * 60   # suspend/resume grace: one full battery eval cycle

EVAL_RE = re.compile(
    r"iter\s+(\d+) \| train (\S+) \| val (\S+) \| lr (\S+) \| (\S+) tok/s"
)


def now():
    return datetime.now().strftime("%F %T")


def wlog(msg):
    try:
        with open(WD_LOG, "a") as f:
            f.write(f"{now()} {msg}\n")
    except OSError:
        pass  # FIX 5: ENOSPC / missing logs/ must never kill the caller,
        # especially the `except Exception` handler at the bottom of the loop


def status(state, detail):
    try:
        with open(STATUS, "w") as f:
            f.write(f"{state} {now()} {detail}\n")
    except OSError:
        pass  # FIX 5: STATUS is best-effort; writing it must never raise


_alert_seen = {}


def alert(key, msg, force=False):
    """Append-only human-visible alert (FIX 8).

    STATUS is a single line the 10-min health check clobbers back to OK, so
    kill/halt/crash-loop/battery/RAM/disk alerts also land in ALERT.log, which
    is never truncated. Persistent conditions are deduped per key for 1h so a
    30s tick cannot spam a line every loop. Never raises (same WHY as FIX 5).
    """
    t = time.time()
    if not force and t - _alert_seen.get(key, 0.0) < 3600:
        return
    _alert_seen[key] = t
    try:
        with open(ALERT_LOG, "a") as f:
            f.write(f"{now()} [{key}] {msg}\n")
    except OSError:
        pass


_tick_fails = 0


def note_tick_failure(exc):
    """ISSUE 3: count consecutive main-loop tick failures and surface them.

    A persistent per-tick exception (e.g. PermissionError reading train.log)
    used to reach only wlog(), leaving ALL detection silently dead for the
    rest of the run with ZERO ALERT.log entries. After 3 in a row, alert()
    (deduped per key for 1h) makes it durable. Never raises (same WHY as
    FIX 5: this runs from the catch-all that must never die).
    """
    global _tick_fails
    _tick_fails += 1
    wlog(f"tick error: {exc}")
    if _tick_fails >= 3:
        alert("tick-error", f"{_tick_fails} consecutive tick failures -- "
                            f"detection degraded: {exc}")


def note_tick_success():
    """ISSUE 3: a fully-completed tick resets the consecutive-failure count."""
    global _tick_fails
    _tick_fails = 0


def train_pids():
    out = subprocess.run(
        ["pgrep", "-f", r"train[.]py --preset"],
        capture_output=True, text=True,
    ).stdout
    return [int(p) for p in out.split()]


class LogReadError(Exception):
    """train.log exists but cannot be read (PermissionError, EIO, ...).

    ISSUE 3: only FileNotFoundError used to be handled, so any other OSError
    bubbled into the main loop's catch-all as an opaque recurring 'tick error'
    with zero ALERT.log coverage -- detection silently dead for the whole
    run. Raised instead so main() surfaces it as an explicit condition.
    """


def current_run_text():
    """Text of train.log since the LAST `=== RUN` header.

    FIX 1: train.log is append-only across runs; slicing from the last header
    means every content check (NaN, divergence, stall, completion) only ever
    sees the current run instead of stale rows from a previous one -- otherwise
    a fresh run at iter 0-39 was judged entirely on the old run's rows and got
    re-killed within one tick, forever.
    """
    try:
        txt = open(TRAIN_LOG, errors="replace").read()
    except FileNotFoundError:
        return ""  # FIX 7: caller detects the missing log and alerts on it
    except OSError as e:
        raise LogReadError(str(e)) from e  # ISSUE 3: explicit, not opaque
    i = txt.rfind("=== RUN")
    return txt[i:] if i >= 0 else txt


def parse_evals(text, n=40):
    rows = []
    for m in EVAL_RE.finditer(text):  # FIX 1: current run only
        try:
            tr = float(m.group(2))
            va = float(m.group(3))
        except ValueError:
            tr = va = float("nan")
        rows.append((int(m.group(1)), tr, va, m.group(5)))
    return rows[-n:]


def last_evals(n=40):
    return parse_evals(current_run_text(), n)


def run_completed(text):
    """ISSUE 5: does THIS run's text contain its own completion line?

    With a `=== RUN` header, current_run_text() already sliced at the last
    one, so any completion line in the slice belongs to the current run.
    Without a header (a manually launched run), train.log still holds older
    runs' text -- including a stale 'training complete.' -- so accept only
    when the completion line is the LAST line written: the current run must
    have printed it itself as its final output. Otherwise a crashed headerless
    run would be marked DONE and left dead: no restart, no alert.
    """
    if "=== RUN" in text:
        return "training complete." in text
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return bool(lines) and "training complete." in lines[-1]


def kill_train(reason, detail, auto_recover=False):
    """Kill training and arm HALT -- unless an operator flag already exists.

    ISSUE 2: HALT is opened O_CREAT|O_EXCL instead of "w". "w" always
    truncated, so a pre-existing OPERATOR flag was destroyed before
    recover_hang() could see it (its own-check reads "halted by watchdog"),
    and the subsequent os.remove violated "armed = never restarts". O_EXCL
    also closes the 30s-tick race: if the operator arms .halt between the
    loop-top read and this kill, creation fails and their flag survives. A
    watchdog-initiated kill still arms a halt when none exists.

    ISSUE 4: auto_recover says whether the halt is about to be dropped again
    immediately (recover_hang), so the alert wording matches reality instead
    of training the operator to ignore every "needs operator" line.
    """
    snap = os.path.join(LOGS, f"alert-{reason}-{datetime.now().strftime('%H%M%S')}.txt")
    try:
        tail = subprocess.run(["tail", "-40", TRAIN_LOG], capture_output=True, text=True).stdout
        with open(snap, "w") as f:
            f.write(f"reason: {reason} -- {detail}\n\n{tail}")
    except Exception:
        pass
    for pid in train_pids():
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    time.sleep(3)
    for pid in train_pids():
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        fd = os.open(HALT, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(fd, "w") as f:
            f.write(f"{now()} halted by watchdog: {reason}\n")
    except FileExistsError:
        # ISSUE 2: an operator flag (pre-existing or armed this same tick)
        # is left byte-for-byte intact; recover_hang()'s own-check then sees
        # text without "halted by watchdog" and refuses to remove it.
        wlog(f"kill_train: pre-existing HALT left untouched ({reason})")
    except OSError:
        pass  # FIX 5: a halt write must never raise out of kill_train
    note = "auto-restarting" if auto_recover else "needs operator"  # ISSUE 4
    # FIX 8: alert() appends so this kill survives the health check clobbering STATUS
    alert("kill", f"training killed ({reason}): {detail} -- {note}", force=True)
    wlog(f"KILLED training ({reason}): {detail} -- snapshot {os.path.basename(snap)}")
    status("ALERT", f"training killed ({reason}): {detail} -- {note}")


def restart(reason):
    if os.path.exists(HALT):
        return
    try:
        stamps = [float(s) for s in open(RESTARTS).read().split()]
    except (FileNotFoundError, ValueError):
        # FIX 2: only FileNotFoundError was caught, so a torn/partial .restarts
        # write (non-atomic, interrupted by kill/disk-full) or a stray token
        # raised ValueError on EVERY tick, swallowed by the outer catch-all ->
        # auto-restart silently disabled forever while STATUS still read OK.
        # Treat unreadable content as empty/reset instead.
        stamps = []
    t = time.time()
    stamps = [s for s in stamps if t - s < 6 * 3600]  # keep 6h of history
    n1h = len([s for s in stamps if t - s < 3600])
    # FIX 3: 3-in-1h never trips when restarts land 25-35 min apart (pruned to
    # a 60-min window), so a slow crash loop restarted forever with ZERO net
    # progress -- the 6h window catches it even when each cycle is slow.
    if n1h >= 3 or len(stamps) >= 6:
        why = f"{n1h} in 1h" if n1h >= 3 else "6 in 6h"
        try:
            with open(HALT, "w") as f:
                f.write(f"{now()} restart loop ({why}), manual intervention needed\n")
        except OSError:
            pass  # FIX 5: never raise while handling a failure
        alert("crash-loop", f"restart loop ({why}) -- halted for operator", force=True)
        wlog(f"RESTART LOOP blocked ({reason}) -- halted")
        status("ALERT", f"crash loop: {why}, halted for operator")
        return
    stamps.append(t)
    try:
        # FIX 2: atomic replace so an interrupt mid-write cannot tear .restarts
        tmp = RESTARTS + ".tmp"
        with open(tmp, "w") as f:
            f.write(" ".join(str(s) for s in stamps))
        os.replace(tmp, RESTARTS)
    except OSError:
        pass  # limiter is best-effort; iter-stall (FIX 4) is the progress gate
    subprocess.Popen(
        LAUNCH, shell=True, executable="/bin/bash",
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    wlog(f"AUTO-RESTART ({reason})")
    status("RESTART", f"auto-restart after {reason}")


def recover_hang(reason, detail):
    """Hang-type failures are recoverable: kill, drop the halt, auto-restart.

    FIX 6b: kill_train used to be followed by `with open(HALT): pass` +
    unconditional os.remove, which could destroy an operator's armed flag and
    violate "armed = never restarts". Now we only ever remove a halt WE wrote
    (kill_train stamps it "halted by watchdog"); if the operator's flag is the
    one present -- kill_train's O_EXCL leaves it untouched (ISSUE 2) -- the
    own-check fails, restart() sees it and stays quiet as documented.
    """
    kill_train(reason, detail, auto_recover=True)  # ISSUE 4: alert says "auto-restarting"
    try:
        own = "halted by watchdog" in open(HALT).read()
    except (FileNotFoundError, OSError):
        own = False
    if own:
        try:
            os.remove(HALT)
        except FileNotFoundError:
            pass
    restart(reason)


def power():
    try:
        base = "/sys/class/power_supply"
        for d in os.listdir(base):
            try:
                if open(f"{base}/{d}/type").read().strip() == "Mains":
                    if open(f"{base}/{d}/online").read().strip() == "1":
                        return "AC"
            except OSError:
                continue
        return "BATTERY"
    except OSError:
        return "?"


def health_line(rows, pids):
    it = tr = va = tps = "-"
    if rows:
        it, tr, va, tps = rows[-1]
    rss = "?"
    if pids:
        try:
            rss = subprocess.run(
                ["ps", "-o", "rss=", "-p", str(pids[0])],
                capture_output=True, text=True,
            ).stdout.strip()
            rss = f"{int(rss) // 1024}MB"
        except Exception:
            pass
    disk = shutil.disk_usage("/").free // 2**30
    line = (f"HEALTH iter={it} train={tr} val={va} tps={tps} "
            f"power={power()} rss={rss} diskfree={disk}G")
    wlog(line)
    return line


def main():
    try:  # FIX 5: guard startup logging -- must never die before entering the loop
        wlog("=== watchdog started ===")
        status("OK", "watchdog armed")
    except Exception:
        pass
    last_health = 0.0
    last_power = None
    ignore_silence_until = 0.0
    prev_tick = time.time()
    last_iter = None        # FIX 4: last observed eval iteration (stall gate)
    iter_stall_since = 0.0  # FIX 4: when that iteration was first observed
    no_eval_since = 0.0     # FIX 7: when parseable eval lines went missing
    while True:
        try:
            # suspend/resume detection: our own sleep overshooting = system slept
            gap = time.time() - prev_tick - 30
            if gap > 120:
                # ISSUE 1: grace covers a full battery-speed eval cycle, and the
                # iter-stall / no-eval budgets are reset too -- wall-clock time
                # spent asleep must not count toward either, or resume would
                # trip a false kill even though the silence grace was extended.
                ignore_silence_until = time.time() + SUSPEND_GRACE_S
                last_iter = None
                iter_stall_since = 0.0
                no_eval_since = 0.0
                wlog(f"suspend/resume detected (+{gap/60:.1f} min gap), "
                     f"{SUSPEND_GRACE_S//60}-min silence grace, stall/no-eval timers reset")
            prev_tick = time.time()

            halted = os.path.exists(HALT)
            pids = train_pids()

            if not pids:
                # FIX 4/7: whatever comes next is a fresh run -- drop the
                # previous run's progress/detection timers so a new run is
                # judged only on its own output, never on stale state
                last_iter = None
                iter_stall_since = 0.0
                no_eval_since = 0.0
                if halted:
                    status("HALTED", open(HALT).read().strip())
                else:
                    try:
                        txt = current_run_text()
                    except LogReadError as e:
                        # ISSUE 3/5: unreadable log = cannot confirm completion
                        # or crash. Do not restart blind and never claim DONE.
                        alert("log-unreadable", f"train.log unreadable: {e} "
                                                f"-- cannot judge dead process")
                        status("ALERT", f"train.log unreadable: {e}")
                        time.sleep(30)
                        continue
                    if run_completed(txt):
                        # FIX 9: normal completion is not a crash -- do not restart
                        # 3x into a permanent "restart loop" halt; go DONE and stay
                        # quiet (still alive so a later relaunch is watched again).
                        # ISSUE 5: run_completed() only accepts a completion line
                        # this run itself printed, so an old run's "training
                        # complete." cannot mask a fresh early crash.
                        alert("done", "training complete -- run finished normally")
                        status("DONE", "training complete -- watchdog idle, still watching")
                    else:
                        restart("process dead")
                time.sleep(30)
                continue

            p = power()
            if last_power is not None and p != last_power:
                if p == "BATTERY":
                    wlog("ALERT: on BATTERY (40% speed loss) -- plug in")
                else:
                    wlog("power restored: AC")
            if p == "BATTERY":
                alert("battery", "training is running ON BATTERY (~40% slower)")
                status("ALERT", "training is running ON BATTERY -- plug in (~40% slower)")
            elif last_power is not None and p != last_power:
                status("OK", "back on AC, full speed")
            last_power = p

            try:
                mem_kb = int(re.search(
                    r"MemAvailable:\s+(\d+)",
                    open("/proc/meminfo").read(),
                ).group(1))
            except Exception:
                mem_kb = 999999
            disk_free = shutil.disk_usage("/").free
            if mem_kb < 150_000:
                alert("ram", f"MemAvailable {mem_kb//1000}MB critical")
                wlog(f"ALERT: MemAvailable {mem_kb//1000}MB")
                status("ALERT", f"RAM critical: {mem_kb//1000}MB available")
            if disk_free < 5 * 2**30:
                alert("disk", "disk <5GB free")
                wlog("ALERT: disk <5GB free")
                status("ALERT", "disk <5GB free")

            log_missing = False
            log_unreadable = None
            try:
                silent = time.time() - os.path.getmtime(TRAIN_LOG)
            except FileNotFoundError:
                silent = 0.0
                log_missing = True  # FIX 7: a missing log is a detection gap, not "fresh"
            except OSError as e:
                silent = 0.0
                log_unreadable = str(e)  # ISSUE 3: stat failed for a non-missing reason
            try:
                rows = last_evals()
            except LogReadError as e:
                rows = []
                log_unreadable = str(e)  # ISSUE 3: explicit condition, not a tick crash
                alert("log-unreadable", f"train.log unreadable: {e} -- detection degraded")

            # FIX 6a: EVERY kill/restart content check below is gated on the
            # operator's halt flag NOT being present -- armed = never kill,
            # never restart, watchdog stays quiet (documented contract).
            if not halted and time.time() > ignore_silence_until:
                if log_missing or not rows:
                    # FIX 7: a renamed/rotated train.log or a regex drift after
                    # a train.py format change used to silently disable ALL
                    # hang/NaN/divergence detection (empty `rows` gated every
                    # check) while STATUS still read OK. Make it loud, and
                    # after 20 min treat it exactly like a hang.
                    if no_eval_since == 0.0:
                        no_eval_since = time.time()
                    ne = time.time() - no_eval_since
                    # ISSUE 1: same cadence budget as the silence gate below --
                    # evals land every ~14-15 min AC / ~20-22 min battery.
                    det = f"log_missing={log_missing} unreadable={log_unreadable}"
                    if ne > SILENT_KILL_S:
                        wlog(f"no parseable evals for {ne/60:.0f}min "
                             f"({det}) -> kill+restart")
                        recover_hang("hang", f"no parseable evals {ne/60:.0f}min "
                                             f"({det})")
                        last_iter = None
                        iter_stall_since = 0.0
                        no_eval_since = 0.0
                        time.sleep(30)
                        continue
                    if ne > SILENT_ALERT_S:
                        alert("no-eval", f"no parseable evals for {ne/60:.0f}min "
                                         f"({det}) -- detection degraded")
                        status("ALERT", f"no parseable evals for {ne/60:.0f}min -- detection degraded")
                else:
                    no_eval_since = 0.0
                    # FIX 4/ISSUE 1: iter-stall gate. eval_interval=12 lands an
                    # eval every ~14-15 min on AC / ~20-22 min on battery, so
                    # STALL_BUDGET_S=90min (~4x the slowest normal cycle) without
                    # the iteration number advancing = no progress (the slow-death
                    # loop of FIX 3: restarts redo the same iteration forever with
                    # ZERO net progress while the log stays lively). Reset on any
                    # change, and on suspend/resume so sleep time is not billed.
                    cur = rows[-1][0]
                    if cur != last_iter:
                        last_iter = cur
                        iter_stall_since = time.time()
                    elif iter_stall_since and time.time() - iter_stall_since > STALL_BUDGET_S:
                        wlog(f"iter {cur} stalled {(time.time()-iter_stall_since)/60:.0f}min "
                             "-> kill+halt")
                        kill_train("stall", f"iter {cur} unchanged for "
                                            f"{(time.time()-iter_stall_since)/60:.0f}min")
                        last_iter = None
                        iter_stall_since = 0.0
                        time.sleep(30)
                        continue

                if rows:
                    if silent > SILENT_KILL_S:  # ISSUE 1: >=2x battery eval cycle
                        wlog(f"log silent {silent/60:.0f}min with live process -> kill+restart")
                        recover_hang("hang", f"log silent {silent/60:.0f}min")
                        last_iter = None
                        iter_stall_since = 0.0
                        no_eval_since = 0.0
                        time.sleep(30)
                        continue
                    if silent > SILENT_ALERT_S:  # ISSUE 1: above battery eval cycle
                        alert("silent", f"log silent {silent/60:.0f}min (process alive?)")
                        status("ALERT", f"log silent {silent/60:.0f}min (process alive?)")

                    # nan / inf anywhere recent (rows are current-run only: FIX 1)
                    bad = [r for r in rows[-6:] if not (math.isfinite(r[1]) and math.isfinite(r[2]))]
                    if bad:
                        kill_train("nan", f"iter {bad[-1][0]} train={bad[-1][1]} val={bad[-1][2]}")
                        time.sleep(30)
                        continue

                    # sustained post-warmup climb with fixed eval windows = real
                    # (r[0] >= 40 now only ever sees THIS run's iters: FIX 1)
                    recent = [r for r in rows if r[0] >= 40][-8:]
                    if len(recent) >= 7:
                        rises = sum(
                            1 for i in range(1, len(recent))
                            if recent[i][2] > recent[i - 1][2]
                        )
                        net = recent[-1][2] - recent[0][2]
                        if rises >= len(recent) - 1 and net >= 0.04:
                            kill_train(
                                "divergence",
                                f"iter {recent[0][0]}->{recent[-1][0]} "
                                f"val {recent[0][2]:.3f}->{recent[-1][2]:.3f} "
                                f"({rises}/{len(recent)-1} rising)",
                            )
                            time.sleep(30)
                            continue

            if time.time() - last_health > 600:
                hl = health_line(rows, pids)
                last_health = time.time()
                if not os.path.exists(HALT):
                    status("OK", hl[len("HEALTH "):])
        except Exception as e:  # watchdog must never die
            note_tick_failure(e)  # ISSUE 3: recurring failures must alert, not fade
            time.sleep(30)
            continue
        note_tick_success()  # ISSUE 3: fully-completed tick clears the counter
        time.sleep(30)


if __name__ == "__main__":
    main()
