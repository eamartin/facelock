from pprint import pprint
import subprocess

def handle_buffer(buff):
    PRESENT_SIZE = 130 # smaller than this and you don't count
    # thats what she said

    frames = {'valid': 0, 'invalid': 0, 'empty': 0}
    total = 0.0
    for frame in buff:
        for (x, y, w, h), valid in frame.iteritems():
            size = float(w + h) / 2.0
            if size < PRESENT_SIZE:
                frames['empty'] += 1
            elif valid:
                frames['valid'] += 1
            else:
                frames['invalid'] += 1

    lock_screen = lambda: subprocess.call(['gnome-screensaver-command', '-l'])
    if frames['empty'] > .8 * len(buff):
        lock_screen()
    if frames['invalid'] > .7 * len(buff):
        # do stuff and then lock screen
        pass
