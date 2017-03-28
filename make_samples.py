# -*- coding: utf-8 -*-

import os
import configurations
import psycopg2.pool


flickr8k_images = []

for seg in configurations.flickr8k_path:
	image_path = os.listdir(seg)
	for path in image_path:
		item = {}
		item['path'] = os.path.join(seg, path)
		item['link'] = "/static/images/flickr8k/" + path
		flickr8k_images.append(item)

flickr30k_images = []

for seg in configurations.flickr30k_path:
	image_path = os.listdir(seg)
	for path in image_path:
		item = {}
		item['path'] = os.path.join(seg, path)
		item['link'] = "/static/images/flickr30k/" + path
		flickr30k_images.append(item)

coco_images = []

for seg in configurations.coco_path:
	image_path = os.listdir(seg)
	for path in image_path:
		item = {}
		item['path'] = os.path.join(seg, path)
		item['link'] = "/static/images/coco/" + seg.split('/')[-2] + '/' + path
		coco_images.append(item)

pool = psycopg2.pool.ThreadedConnectionPool(5, 10, host = '127.0.0.1', port = '5432', database = 'image', user = 'ffq', password = '123456')

conn = pool.getconn()

cur = conn.cursor()

for item in flickr8k_images:
	cur.execute("INSERT INTO samples (link, path) VALUES (%s, %s)", (item['link'], item['path']))

for item in flickr30k_images:
	cur.execute("INSERT INTO samples (link, path) VALUES (%s, %s)", (item['link'], item['path']))

for item in coco_images:
	cur.execute("INSERT INTO samples (link, path) VALUES (%s, %s)", (item['link'], item['path']))

conn.commit()

cur.close()

conn.close()

pool.closeall()