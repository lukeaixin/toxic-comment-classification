import sys
import time

from contextlib import contextmanager

def seconds_to_str(seconds):
    seconds = int(seconds) + 1
    minutes, secs = divmod(seconds, 60)
    hours, mins = divmod(minutes, 60)
    days, hrs = divmod(hours, 24)
    converted_time = [(days, 'days'), (hrs, 'hrs'), (mins, 'mins'), (secs, 'secs')]
    return ', '.join('{} {}'.format(val, unit) for val, unit in converted_time if val)

def log_message(message):
    print(message)
    sys.stdout.flush()

@contextmanager
def log_runtime(message):
    log_message(message)
    start = time.time()
    try:
        yield
    finally:
        end = time.time()
        duration = seconds_to_str(end - start)
        log_message('Total time spent: {}'.format(duration))
