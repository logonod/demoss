#-*- coding: utf-8 -*-
import pandas as pd
import os
import json
from collections import namedtuple
import random
import numpy as np
from util import load_image
import skimage.io
import matplotlib.pyplot as plt
import os
import cPickle
import matplotlib.pyplot as plt


flickr8k_path = [{"images": "./images/flickr8k/",
                  "captions": "./flickr8kzhc.caption.txt"}]

flickr8k_info = []

for path in flickr8k_path:
    t = pd.read_table(path["captions"], sep=' ', header=None, names=['image', 'caption'])
    t['image_num'] = t['image'].map(lambda x: x.split('#')[1])
    t['image'] = t['image'].map(lambda x: os.path.join(path['images'], x.split('#')[0]))
    for i in t.groupby(['image']):
        d = {}
        d['image'] = i[0]
        d['captions'] = i[1]['caption']
        flickr8k_info.append(d)

for d in flickr8k_info[:100]:
	image = load_image(d['image'])
	plt.imshow(image) 
	plt.show()
	for i, c in enumerate(d['captions'].values):
		print "ref", str(i+1) + ":", c