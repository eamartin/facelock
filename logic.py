import base64
import json
import os
from pprint import pprint
import subprocess

import cv
import requests

from settings import *

EMAIL_URL = 'http://anonymouse.org/cgi-bin/anon-email.cgi'
MY_EMAIL = '8327971548@messaging.sprintpcs.com'
INTRUDER_DIR = ROOT_DIR + 'intruders/'
IMGUR_URL = 'http://api.imgur.com/2/upload.json'
IMGUR_KEY = '00c8de2c75e534adbf7d5497a8397a88'

def handle_buffer(buff, image):
    PRESENT_SIZE = 130 # smaller than this and you don't count
    # thats what she said

    frames = {'valid': 0, 'invalid': 0, 'empty': 0}
    total = 0.0
    for frame in buff:
        if frame == {}:
            frames['empty'] += 1
            continue
        for (x, y, w, h), valid in frame.iteritems():
            size = float(w + h) / 2.0
            if size < PRESENT_SIZE:
                frames['empty'] += 1
            elif valid:
                frames['valid'] += 1
            else:
                frames['invalid'] += 1

    lock_screen = lambda: subprocess.call(['gnome-screensaver-command', '-l'])

    if frames['valid'] > .8 * len(buff):
        subprocess.call(['gnome-screensaver-command', '-d'])
        return

    if frames['empty'] > .5 * len(buff) and frames['valid'] < .1 * len(buff):
        lock_screen()

    if frames['invalid'] > .65 * len(buff):
        lock_screen()
        pic_path = INTRUDER_DIR + 'intruder.jpg'
        cv.SaveImage(pic_path, image)
        with open(pic_path, 'rb') as f:
            data = f.read()

        imgur = {
            'key': IMGUR_KEY,
            'image': base64.b64encode(data)
        }
        try:
            r = requests.post(IMGUR_URL, data=imgur)
            url = r.json['upload']['links']['imgur_page']

            email = {
                'to': MY_EMAIL,
                'subject': 'Laptop Intruder',
                'text': url
            }
            requests.post(EMAIL_URL, data=email)
        except Exception as e:
            print 'Notification failed'
            print e

        os.system('mpg321 ' + ROOT_DIR + 'siren.mp3')
