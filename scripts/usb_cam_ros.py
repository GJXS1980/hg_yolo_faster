#! /usr/bin/env python3
# -*- coding:utf-8 -*-

import  rospy
import os
import cv2
import time
import argparse

import torch
import model.detector

import utils
from cv_bridge import CvBridge,  CvBridgeError
from sensor_msgs.msg import Image
from hg_yolo_faster.msg import yolo_label



if __name__ == '__main__':
    #   初始化ROS节点
    rospy.init_node("yolo_cam", anonymous=True)

    bridge = CvBridge()
    image_pub = rospy.Publisher("cv_beidge_image", Image, queue_size=1)
    data_pub = rospy.Publisher("yolo_data", yolo_label, queue_size=1)
    yolo_data = yolo_label()

    #指定训练配置文件
    model_data = rospy.get_param("~model_data", "/home/grantli/data_ws/Yolo-FastestV2/data/coco.data")
    model_weights = rospy.get_param("~model_weights", "/home/grantli/data_ws/Yolo-FastestV2/modelzoo/coco2017-0.241078ap-model.pth")

    cfg = utils.load_datafile(model_data)
    # print(cfg)

    #模型加载
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = model.detector.Detector(cfg["classes"], cfg["anchor_num"], True).to(device)
    model.load_state_dict(torch.load(model_weights, map_location=device))

    #sets the module in eval node
    model.eval()
    
    video_cam = int(rospy.get_param("~video_cam", 0))
    cap = cv2.VideoCapture(video_cam)
    if not cap.isOpened():
        raise ValueError("Video open failed")
        # return
    status = True
    sum_time = 0.0 #当前耗费总时长
    fps_curr = 0.0 #当前帧率
    frame_num = 0  #当前读入总帧数

    while not rospy.is_shutdown() and status:
        starttime = time.time()
        frame_num = frame_num +1

        status, ori_img = cap.read()
        if (status):
            #数据预处理
            res_img = cv2.resize(ori_img, (cfg["width"], cfg["height"]), interpolation = cv2.INTER_LINEAR) 
            img = res_img.reshape(1, cfg["height"], cfg["width"], 3)
            img = torch.from_numpy(img.transpose(0,3, 1, 2))
            img = img.to(device).float() / 255.0

            #模型推理
            # start = time.perf_counter()
            preds = model(img)
            # end = time.perf_counter()
            # time = (end - start) * 1000.
            # print("forward time:%fms"%time)

            #特征图后处理
            output = utils.handel_preds(preds, cfg, device)
            output_boxes = utils.non_max_suppression(output, conf_thres = 0.3, iou_thres = 0.4)

            #加载label names
            LABEL_NAMES = []
            with open(cfg["names"], 'r') as f:
                for line in f.readlines():
                    LABEL_NAMES.append(line.strip())
            
            h, w, _ = ori_img.shape
            scale_h, scale_w = h / cfg["height"], w / cfg["width"]

            #绘制预测框
            for box in output_boxes[0]:
                box = box.tolist()
            
                obj_score = box[4]
                category = LABEL_NAMES[int(box[5])]

                x1, y1 = int(box[0] * scale_w), int(box[1] * scale_h)
                x2, y2 = int(box[2] * scale_w), int(box[3] * scale_h)

                if obj_score > 0.85:
                    cv2.rectangle(ori_img, (x1, y1), (x2, y2), (255, 255, 0), 2)
                    cv2.putText(ori_img, '%.2f' % obj_score, (x1, y1 - 5), 0, 0.7, (0, 255, 0), 2)	
                    cv2.putText(ori_img, category, (x1, y1 - 25), 0, 0.7, (0, 255, 0), 2)

                    #   求中心点坐标
                    # print(category, obj_score, ((x2 - x1)/2), (y2 - y1)/2)
                    yolo_data.category = category
                    yolo_data.obj_score = obj_score
                    yolo_data.x = (x2 - x1)/2
                    yolo_data.y = (y2 - y1)/2
                    data_pub.publish(yolo_data)

            #   计算帧率
            # fps  = cap.get(cv2.CAP_PROP_FPS)
            endtime = time.time()
            fps = (endtime - starttime)*1000
            cv2.putText(ori_img, ("fps:" + str(fps)), (10, 20), 0, 0.7, (0, 255, 0), 2)

            #   显示图像
            cv2.imshow("ori_img", ori_img)
            cv2.waitKey(10)
            #   发布识别图形
            image_pub.publish(bridge.cv2_to_imgmsg(ori_img))

