#-*- coding: utf-8 -*-
import matplotlib.pyplot as plt

import cPickle


data = cPickle.load(open("data.pkl","rb"))

for x in data:
	plt.imshow(x['image'])
	plt.show()
	print "predicted:", x['pred']
	for i, c in enumerate(x['captions']):
		print "ref", str(i+1) + ":", c