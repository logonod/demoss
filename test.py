#-*- coding: utf-8 -*-
import pandas as pd
import os
import json
from collections import namedtuple
from cnn_util import *
import random
import numpy as np
import model
import tensorflow as tf
from util import load_image
import skimage.io
import matplotlib.pyplot as plt
import os
import cPickle



pickles = []


def test(test_feat='./Cacoustic-guitar-player.npy', model_path='./models/model-99', maxlen=30, image_paths=None, captions=None): # Naive greedy search

    ixtoword = np.load('data/ixtoword.npy').tolist()
    n_words = len(ixtoword)

    feats = np.load(test_feat)
    
    
    # 이 부분이 존나 중요함. 계속 caption_generator를 가져온 뒤 바로 restore를 했었는데,
    # TensorFlow의 LSTM은 call을 한 뒤에 weight가 만들어지기 때문에 build_generator보다 뒤쪽에서 restore를 해야 함.

    
    for feat, image_path, caption in zip(feats, image_paths, captions):
        pickle = {}
        pickle['image'] = load_image(image_path) 
        caption_generator = model.Caption_Generator(
           dim_image=model.dim_image,
           dim_hidden=model.dim_hidden,
           dim_embed=model.dim_embed,
           batch_size=model.batch_size,
           n_lstm_steps=maxlen,
           n_words=n_words)

        image, generated_words = caption_generator.build_generator(maxlen=maxlen)
        sess = tf.InteractiveSession()
        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        feat = [feat]

        generated_word_index= sess.run(generated_words, feed_dict={image:feat})
        generated_word_index = np.hstack(generated_word_index)

        generated_sentence = [ixtoword[x] for x in generated_word_index]

        punctuation = np.argmax(np.array(generated_sentence) == '.')+1

        generated_words = generated_sentence[:punctuation]
        generated_sentence = ' '.join(generated_words)
        pickle['pred'] = generated_sentence
        pickle['captions'] = caption
        pickles.append(pickle)
        print generated_sentence
        sess.close()
        tf.reset_default_graph()
        #break




flickr8k_path = [{"images": "./images/flickr8k/",
                  "captions": "./captions/flickr8k/Flickr8k.token.txt"}]

flickr30k_path = [{"images": "./images/flickr30k/",
                   "captions": "./captions/flickr30k/results_20130124.token"}]

coco_path = [{"images": "./images/coco/train2014/",
              "captions": "./captions/coco/captions_train2014.json"},
             {"images": "./images/coco/val2014/",
              "captions": "./captions/coco/captions_val2014.json"}]

flickr8k_info = []

for path in flickr8k_path:
    t = pd.read_table(path["captions"], sep='\t', header=None, names=['image', 'caption'])
    t['image_num'] = t['image'].map(lambda x: x.split('#')[1])
    t['image'] = t['image'].map(lambda x: os.path.join(path['images'], x.split('#')[0]))
    for i in t.groupby(['image']):
        d = {}
        d['image'] = i[0]
        d['captions'] = i[1]['caption']
        flickr8k_info.append(d)

flickr30k_info = []

for path in flickr30k_path:
    t = pd.read_table(path["captions"], sep='\t', header=None, names=['image', 'caption'])
    t['image_num'] = t['image'].map(lambda x: x.split('#')[1])
    t['image'] = t['image'].map(lambda x: os.path.join(path['images'], x.split('#')[0]))
    for i in t.groupby(['image']):
        d = {}
        d['image'] = i[0]
        d['captions'] = i[1]['caption']
        flickr30k_info.append(d)

coco_info = []

ImageMetadata = namedtuple("ImageMetadata",
                           ["image_id", "filename", "captions"])

for path in coco_path:
    with open(path["captions"]) as f:
        caption_data = json.load(f)
    id_to_filename = [(x["id"], x["file_name"]) for x in caption_data["images"]]
    id_to_captions = {}
    for annotation in caption_data["annotations"]:
        image_id = annotation["image_id"]
        caption = annotation["caption"]
        id_to_captions.setdefault(image_id, [])
        id_to_captions[image_id].append(caption)
    for image_id, base_filename in id_to_filename:
        filename = os.path.join(path["images"], base_filename)
        captions = [c for c in id_to_captions[image_id]]
        coco_info.append(ImageMetadata(image_id, filename, captions))

vgg_model = '/home/ffq/2017-caption/caffe/models/vgg/VGG_ILSVRC_19_layers.caffemodel'
vgg_deploy = '/home/ffq/2017-caption/caffe/models/vgg/VGG_ILSVRC_19_layers_deploy.prototxt'

cnn = CNN(model=vgg_model, deploy=vgg_deploy, width=224, height=224)

random.shuffle(flickr8k_info, random.random)

flickr8k_test = flickr8k_info[:30]

flickr8k_test_paths = [x['image'] for x in flickr8k_test]

flickr8k_test_caps = [list(x['captions'].values) for x in flickr8k_test]

feats = cnn.get_features(flickr8k_test_paths)

np.save("./feats.npy", feats)

test("./feats.npy", image_paths = flickr8k_test_paths, captions= flickr8k_test_caps)

random.shuffle(flickr30k_info, random.random)

flickr30k_test = flickr30k_info[:30]

flickr30k_test_paths = [x['image'] for x in flickr30k_test]

flickr30k_test_caps = [list(x['captions'].values) for x in flickr30k_test]

feats = cnn.get_features(flickr30k_test_paths)

np.save("./feats.npy", feats)

test("./feats.npy", image_paths = flickr30k_test_paths, captions= flickr30k_test_caps)

random.shuffle(coco_info, random.random)

coco_test = coco_info[:30]

coco_test_paths = [x.filename for x in coco_test]

coco_test_caps = [x.captions for x in coco_test]

feats = cnn.get_features(coco_test_paths)

np.save("./feats.npy", feats)

test("./feats.npy", image_paths = coco_test_paths, captions= coco_test_caps)

cPickle.dump(pickles,open("data.pkl","wb")) 
