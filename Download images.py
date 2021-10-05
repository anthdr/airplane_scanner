#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#import packages
import os
import pandas as pd
from sklearn.model_selection import KFold
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from numpy import asarray

from bing_image_downloader import downloader


# In[ ]:


#make directories to store pictures
get_ipython().system('mkdir mili')
get_ipython().system('mkdir "mili/su-37"')
get_ipython().system('mkdir "mili/f-22"')
get_ipython().system('mkdir "mili/mirage-2000"')
get_ipython().system('mkdir "mili/rafale"')

get_ipython().system('mkdir civi')
get_ipython().system('mkdir "civi/727"')
get_ipython().system('mkdir "civi/707"')
get_ipython().system('mkdir "civi/380"')
get_ipython().system('mkdir "civi/320"')


# In[ ]:


#download pictures from bing in previously created directories
downloader.download("su-37 airplane", limit=600,  output_dir='mili/su-37', adult_filter_off=True, force_replace=False, timeout=60, verbose=True)
downloader.download("f-22 airplane", limit=600,  output_dir='mili/f-22', adult_filter_off=True, force_replace=False, timeout=60, verbose=True)
downloader.download("mirage 2000 airplane", limit=600,  output_dir='mili/mirage-2000', adult_filter_off=True, force_replace=False, timeout=60, verbose=True)
downloader.download("rafale airplane", limit=600,  output_dir='mili/rafale', adult_filter_off=True, force_replace=False, timeout=60, verbose=True)

downloader.download("boeing 727 airplane", limit=600,  output_dir='civi/727', adult_filter_off=True, force_replace=False, timeout=60, verbose=True)
downloader.download("boeing 707 airplane", limit=600,  output_dir='civi/707', adult_filter_off=True, force_replace=False, timeout=60, verbose=True)
downloader.download("airbus 380 airplane", limit=600,  output_dir='civi/380', adult_filter_off=True, force_replace=False, timeout=60, verbose=True)
downloader.download("airbus 320 airplane", limit=600,  output_dir='civi/320', adult_filter_off=True, force_replace=False, timeout=60, verbose=True)


# In[ ]:


#convert all images into .jpg
def convert(zmodel, zairplane, zpath):
    corepath = "C:/Users/antoi/Google Drive/Computer Vision/airplane scanner/" + zmodel + "/" + zairplane + "/" + zpath + "/"
    arr = os.listdir(corepath)
    path = []
    for i in range(len(arr)):
        path.append(corepath + arr[i])
    
    #nested if loop to handle .gif
    for i in range(len(path)):
        if ".gif" in path[i]:
            print("found .gif!")
            cap = cv2.VideoCapture(path[i])
            ret, image = cap.read()
            cap.release()
            replace_path = path[i].replace(".gif", ".jpg")
            #print(replace_path)
            cv2.imwrite(replace_path, image)
        else:
            #print(path[i])
            replace_path = path[i]
            replace_path = replace_path.replace(".png", ".jpg")
            replace_path = replace_path.replace(".jpeg", ".jpg")
            replace_path = replace_path.replace(".JPEG", ".jpg")
            #print(replace_path)
            img = cv2.imread(path[i])
            cv2.imwrite(replace_path, img)
    
    for i in range(len(path)):
        if ".gif" in path[i]:
            os.remove(path[i])
        if ".png" in path[i]:
            os.remove(path[i])
        if ".jpeg" in path[i]:
            os.remove(path[i])
        if ".JPEG" in path[i]:
            os.remove(path[i]) 
        
    print(zairplane, 'done!')


# In[ ]:


convert('civi', '320', 'airbus 320 airplane')
convert('civi', '380', 'airbus 380 airplane')
convert('civi', '707', 'boeing 707 airplane')
convert('civi', '727', 'boeing 727 airplane')

convert("mili", "su-37", "su-37 airplane")
convert("mili", "f-22", "f-22 airplane")
convert("mili", "mirage-2000", "mirage 2000 airplane")
convert("mili", "rafale", "rafale airplane")


# In[ ]:


#make all pictures into 256*256 dimension
pxl = 256

def modify_and_save(zmodel, zairplane, zpath):
    corepath = "C:/Users/antoi/Google Drive/Computer Vision/airplane scanner/" + zmodel + "/" + zairplane + "/" + zpath + "/"
    arr = os.listdir(corepath)
    path = []
    for i in range(len(arr)):
        path.append(corepath + arr[i])


    for i in range(len(path)):
        img = cv2.imread(path[i])
        img = cv2.resize(img, (pxl, pxl), interpolation = cv2.INTER_AREA)
        cv2.imwrite(path[i], img)
        
    print(zairplane, 'done!')


# In[ ]:


modify_and_save('civi', '320', 'airbus 320 airplane')
modify_and_save('civi', '380', 'airbus 380 airplane')
modify_and_save('civi', '707', 'boeing 707 airplane')
modify_and_save('civi', '727', 'boeing 727 airplane')

modify_and_save("mili", "su-37", "su-37 airplane")
modify_and_save("mili", "f-22", "f-22 airplane")
modify_and_save("mili", "mirage-2000", "mirage 2000 airplane")
modify_and_save("mili", "rafale", "rafale airplane")

