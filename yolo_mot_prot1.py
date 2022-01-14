#!/usr/bin/env python                                      
#  -*- coding: utf-8 -*-                                     
import rospy
from darknet_ros_msgs.msg import BoundingBoxes
from sensor_msgs.msg import Image
import time
import math
import numpy as np
import cv2
from cv_bridge import CvBridge

count = 0
limit_count = 0


class Objects_detection():
    def __init__(self):
        rospy.init_node('ar_tracking')
        rospy.loginfo('activate')
        self.objects = [{'xmin':0,
                        'ymin':0,
                        'xmax':0,
                        'ymax':0,
                        'ob_id':0,
                        'undetected': 0,
                        'inheretance':False}] #initializing self.objedts
        self.id_list = list(range(1,20,1))
        self.ob_group = []
        for i ,item in enumerate(self.id_list):
            self.ob_group.append({'id':item,
                                  'contents':[],
                                  'count':0})
        self.detection_sub = rospy.Subscriber('/darknet_ros/bounding_boxes', BoundingBoxes, self.detection_cb)
        self.image_sub = rospy.Subscriber('/Sensor/RealSense435i/color/image_raw', Image, self.image_cb) 

        self.image_pub = rospy.Publisher('tracking_objects', Image, queue_size=10)
    def detection_cb(self, msg):
        bb = msg.bounding_boxes
        for i, item in enumerate(self.objects):
            self.objects[i]['undetected'] += 1
        if bb:
            for i in range(len(bb)):
                if bb[i].Class == 'person':
                    x1 = bb[i].xmax
                    y1 = bb[i].ymax
                    p1 = np.array([x1, y1])
                    for j in range(len(self.objects)):
                        x2 = self.objects[j]['xmax']
                        y2 = self.objects[j]['ymax']
                        p2 = np.array([x2,y2])
                        d = np.linalg.norm(p1 - p2)
                        rate = d/abs(bb[i].xmax - bb[i].xmin)
                        if rate < 2:
                            self.objects[j]['xmax'] = x1
                            self.objects[j]['ymax'] = y1
                            self.objects[j]['xmin'] = bb[i].xmin
                            self.objects[j]['ymin'] = bb[i].ymin
                            self.objects[j]['undetected'] = 0
                            self.objects[j]['inheretance'] = True
                            break
                    new_object = {'xmin': bb[i].xmin,
                                'ymin': bb[i].ymin,
                                'xmax': x1,
                                'ymax': y1,
                                'ob_id': self.id_list.pop(0),
                                'undetected':0,
                                'inheretance':False}
                    self.objects.append(new_object)
        
        for i,item in enumerate(self.objects):
            if item['undetected'] >= 5:
                self.id_list.append(item['ob_id'])
                del self.objects[i]
        
        rospy.loginfo(self.id_list)
        for i, item1 in enumerate(self.ob_group):
            self.ob_group[i]['count'] += 1
            if self.ob_group[i]['count'] >=5:
                self.ob_group[i]['count'] = 0
                self.ob_group[i]['contents'] = []
            for j, item2 in enumerate(self.objects):
                if item1['id'] == item2['ob_id']:
                    self.ob_group[i]['contents'].append( [item2['xmax'], item2['ymax'], item2['xmin'], item2['ymin']] )
                    self.ob_group[i]['count'] = 0
    def image_cb(self, msg):
        bridge = CvBridge()
        frame = bridge.imgmsg_to_cv2(msg,"bgr8") #opencv用に変換
        for i , item in enumerate(self.objects):
            xmax = item['xmax']
            ymax = item['ymax']
            xmin = item['xmin']
            ymin = item['ymin']
            frame = cv2.circle(frame,((xmax+xmin)/2, (ymax+ymin)/2) ,5,(0, 255, 0),1)
        imgMsg = bridge.cv2_to_imgmsg(frame,"bgr8")
        self.image_pub.publish(imgMsg)
                        

if __name__=="__main__":
    try:
        ob = Objects_detection()
        rospy.spin()
    except rospy.ROSInterruptException: pass