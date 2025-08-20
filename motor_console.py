# motor_console.py
import sys

print("Motor console ready. Press Ctrl+C here to close.", flush=True)
for line in sys.stdin:
    sys.stdout.write(line)
    sys.stdout.flush()
