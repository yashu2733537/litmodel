#!/bin/bash
# Safe watchdog (re)launcher. Invoked as "bash launch_watchdog.sh" so no
# pkill/pgrep pattern below can ever match THIS shell's command line.
pkill -f 'watchdog[.]py' 2>/dev/null
sleep 1
cd /home/eneru/litmodel || exit 1
nohup setsid .venv/bin/python watchdog.py > logs/watchdog.out 2>&1 < /dev/null &
sleep 4
if pgrep -f 'watchdog[.]py' >/dev/null; then echo "watchdog running  OK"; else echo "WATCHDOG FAILED"; cat logs/watchdog.out; fi

# reboot persistence: watchdog comes back on boot and auto-restarts training
if command -v crontab >/dev/null; then
  { crontab -l 2>/dev/null | grep -v watchdog.py; echo "@reboot /home/eneru/litmodel/.venv/bin/python /home/eneru/litmodel/watchdog.py >/dev/null 2>&1"; } | crontab -
  echo "crontab @reboot  OK"
fi

echo "--- STATUS:"
cat logs/STATUS
echo "--- training:"
pgrep -f 'train[.]py --preset' >/dev/null && echo "train process ALIVE" || echo "TRAIN PROCESS DEAD"
grep "iter " logs/train.log | tail -3
