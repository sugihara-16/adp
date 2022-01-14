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
        self.ttc = []
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
                            if self.objects[j]['inheretance']:
                                self.objects[j]['xmax'] = (x1+x2)/2
                                self.objects[j]['ymax'] = (y1+y2)/2
                                self.objects[j]['xmin'] = (bb[i].xmin+self.objects[j]['xmin'])/2
                                self.objects[j]['ymin'] = (bb[i].ymin+self.objects[j]['ymin'])/2
                                self.objects[j]['undetected'] = 0
                                self.objects[j]['inheretance'] = True
                            else:                               
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
        #一定時間認識されなかった物体は、追跡対象から外す
        for i,item in enumerate(self.objects):
            if item['undetected'] >= 5:
                self.id_list.append(item['ob_id'])
                del self.objects[i]
        
        change = True
        while(change):
            change = False
            for i,item1 in enumerate(self.objects):
                for j, item2 in enumerate(self.objects):
                    if i == j:
                        continue
                    x1_max = item1['xmax']
                    y1_max = item1['ymax']
                    x1_min = item1['xmin']
                    y1_min = item1['ymin']
                    id_i = item1['ob_id']
                    cx1 = (x1_max+x1_min)/2
                    cy1 = (y1_max+y1_min)/2
                    w1 = abs(x1_max-x1_min)
                    
                    x2_max = item2['xmax']
                    y2_max = item2['ymax']
                    x2_min = item2['xmin']
                    y2_min = item2['ymin']
                    id_j = item2['ob_id']
                    cx2 = (x2_max+x2_min)/2
                    cy2 = (y2_max+y2_min)/2

                    dx = abs(cx1-cx2)
                    #二点の中心間距離がbounding_boxの幅より小さい場合は二点をまとめる(j番目をi番目にくっつける)
                    if dx <= w1:
                        change = True
                        inh = False
                        if self.objects[i]['inheretance']:
                            new_id = id_i
                            self.id_list.append(id_j)
                            inh = True
                        elif self.objects[j]['inheretance']:
                            new_id = id_j
                            self.id_list.append(id_i)
                        else:
                            new_id = id_i
                            self.id_list.append(id_j)
                            inh = True
                        new_object = {'xmin': (x1_min+x2_min)/2,
                            'ymin': max([y1_min,y2_min]),
                            'xmax': (x1_max+x2_max)/2,
                            'ymax': max([y1_max, y2_max]),
                            'ob_id': new_id,
                            'undetected':0,
                            'inheretance':inh}
                        
                        for k, item3 in enumerate(self.objects):
                            if item3['ob_id'] == id_i:
                                del self.objects[k]
                                break
                        for k, item3 in enumerate(self.objects):
                            if item3['ob_id'] == id_j:
                                del self.objects[k]
                                break
                        self.objects.append(new_object)
                        break
                else:
                    continue
                break
        
        for i, item1 in enumerate(self.ob_group):
            self.ob_group[i]['count'] += 1

            if self.ob_group[i]['count'] >=10:
                self.ob_group[i]['count'] = 0
                self.ob_group[i]['contents'] = []
            for j, item2 in enumerate(self.objects):
                if item1['id'] == item2['ob_id']:
                    self.ob_group[i]['contents'].append( [item2['xmax'], item2['ymax'], item2['xmin'], item2['ymin']] )
                    self.ob_group[i]['count'] = 0 
        for i, item in enumerate(self.objects):
            self.objects[i]['inheretance'] = False

    def TTC(self):
        for i, item in enumerate(self.ttc):
            self.ttc[i][4] = False
        for i, item in enumerate(self.ob_group):
            if len(item['contents']) >= 20:
                item_id = item['id']
                arg = -1
                pre_t = 30
                for j, item2 in enumerate(self.ttc):
                    if item2[3] == item_id:
                        arg = j
                        pre_t = self.ttc[j][2]
                        self.ttc[j][4] = True
                        break
                                                        
                x_max_new = item['contents'][len(item['contents'])-1][0]
                x_min_new = item['contents'][len(item['contents'])-1][2] 
                x_max_old = item['contents'][len(item['contents'])-20][0] 
                x_min_old = item['contents'][len(item['contents'])-20][2]
                x_ave_new = (x_max_new+x_min_new)/2

                y_max_new = item['contents'][len(item['contents'])-1][1]
                y_min_new = item['contents'][len(item['contents'])-1][3] 
                y_max_old = item['contents'][len(item['contents'])-20][1] 
                y_min_old = item['contents'][len(item['contents'])-20][3]
                y_new = y_max_new - y_min_new
                y_old = y_max_old - y_min_old 
                y_ave_new = (y_max_new+y_min_new)/2
                u = (y_old - y_new)*30.0/20.0 # u[m/s]
                if u == 0:
                    break
                t = -math.floor(y_new/u * (10 ** 2))/(10**2)
                if arg < 0: 
                    self.ttc.append([x_ave_new,y_ave_new,t,item_id,True])
                if (0 <= arg) :
                    self.ttc[arg] = [x_ave_new,y_ave_new,t,item_id,True]
        for i, item in enumerate(self.ttc):
            if not item[4]:
                del self.ttc[i]

        
    
    def image_cb(self, msg):
        bridge = CvBridge()
        frame = bridge.imgmsg_to_cv2(msg,"bgr8") #opencv用に変換
        
        """
        #追跡物体に円を描画
        for i , item in enumerate(self.objects):
            xmax = item['xmax']
            ymax = item['ymax']
            xmin = item['xmin']
            ymin = item['ymin']
            frame = cv2.circle(frame,((xmax+xmin)/2, (ymax+ymin)/2) ,5,(0, 255, 0),2)
        """
        # 追跡物体のTTCを計算
        self.TTC()
        
        for i, item in enumerate(self.ttc):
            frame = cv2.putText(frame, str(item[2]),(item[0],item[1]),cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0),2 )
        
        """
        for i,item in enumerate(self.ob_group):
            if item['contents']:

                frame = cv2.putText(frame, str(item['id']),(item['contents'][len(item['contents'])-1][0] , item['contents'][len(item['contents'])-1][1]),cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0),2 )

        #追跡物体の軌道を描画
        for i , item1 in enumerate(self.ob_group):
            pre_xmax = pre_ymax = pre_xmin = pre_ymin = None
            for j, item2 in enumerate(self.ob_group[i]['contents']):
                if j == 0:
                    pre_xmax = item2[0]
                    pre_ymax = item2[1]
                    pre_xmin = item2[2]
                    pre_ymin = item2[3]
                    continue
                new_xmax = item2[0]
                new_ymax = item2[1]
                new_xmin = item2[2]
                new_ymin = item2[3]
                frame = cv2.line(frame, 
                                ( (pre_xmax+pre_xmin)/2, (pre_ymax+pre_ymin)/2 ), 
                                ( (new_xmax+new_xmin)/2, (new_ymax+new_ymin)/2 ),
                                (0,255,0),
                                2)
                pre_xmax = new_xmax
                pre_xmin = new_xmin
                pre_ymax = new_ymax
                pre_ymin = new_ymin
        
        """
        #追跡軌道を非線形回帰
        for i,item1 in enumerate(self.ob_group):
            x = []
            y = []
            if not item1['contents']:
                continue
            for j , item2 in enumerate(self.ob_group[i]['contents']):
                x.append( ( item2[0]+item2[2] )/2 )
                y.append( ( item2[1]+item2[3] )/2 )
            #a1,a2,a3,b = np.polyfit(x,y,3)
            a2,a3,b = np.polyfit(x,y,2)
            x_axis = np.arange(np.min(x), np.max(x), 1)
            for k, ele_x in enumerate(x_axis):
                if k == 0:
                    continue
                """
                frame = cv2.line(frame, 
                                ( int(x_axis[k]), int(a1*x_axis[k]**3 + a2*x_axis[k]**2+a3*x_axis[k]+b) ), 
                                ( int(x_axis[k-1]), int(a1*x_axis[k-1]**3 + a2*x_axis[k-1]**2+a3*x_axis[k-1]+b) ),
                                (0,255,0),
                                2)
                """
           
                frame = cv2.line(frame, 
                                ( int(x_axis[k]), int(a2*x_axis[k]**2+a3*x_axis[k]+b) ), 
                                ( int(x_axis[k-1]), int(a2*x_axis[k-1]**2+a3*x_axis[k-1]+b) ),
                                (0,255,0),
                                2)
                
            for j, item2 in enumerate(self.ttc):
                if item2[2] < 0:
                    continue
                if item2[3] == item1['id']:
                    t = item2[2] 
                    v = (x[len(x)-1] - x[0]) / ((1/30.0)*5*len(x))
                    rospy.loginfo(v)
                    #x_f = x[len(x)-1] + v * t
                    x_f = x[len(x) - 1] + 180 + v*0.5
                    #y_f =  int(a2*x_f**2+a3*x_f+b)
                    y_f = y[0]
                    frame = cv2.circle(frame,(int(x_f),int(y_f)) ,5,(0, 0, 255),10)

        imgMsg = bridge.cv2_to_imgmsg(frame,"bgr8")
        self.image_pub.publish(imgMsg)
                        
if __name__=="__main__":
    try:
        ob = Objects_detection()
        rospy.spin()
    except rospy.ROSInterruptException: pass