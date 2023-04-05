#!/usr/bin/python3
import rclpy
from rclpy.node import Node
import cv2
import torch
from torch import hub
import numpy as np
from time import time
from sensor_msgs.msg import Image
from std_msgs.msg import Int8
import pafy
from cv_bridge import CvBridge 
from std_msgs.msg import String

class ObjectDetection(Node):

    def __init__(self):

        self.current_frame = np.zeros((480,640,3))
        self.depth_frame = np.zeros((480,640))
        super().__init__('node_name') 
        self.br = CvBridge()

        self.model = self.load_model()
        self.classes = self.model.names
        if torch.cuda.is_available():
            self.device = 'cuda' 
        else:
            self.device ='cpu'

        print(self.device)

        # self.Image_subscription = self.create_subscription(             
        # Image,                                                     
        # '/zedm/zed_node/rgb/image_rect_color', 
        # self.Image_listener_callback, 
        # 10)
        # self.Image_subscription

        self.Image_subscription = self.create_subscription(             
        Image,                                                     
        '/camera/color/image_raw', 
        self.Image_listener_callback, 
        10)
        self.Image_subscription 

        self.Depth_Image_subscription = self.create_subscription(       
        Image,                                                     
        '/camera/depth/image_rect_raw', 
        self.Depth_Image_listener_callback, 
        10)
        self.Depth_Image_subscription 

    def get_video(self):
        return cv2.VideoCapture(0)

    def Image_listener_callback(self, msg:Image):
        start_time = time()
        self.current_frame = self.br.imgmsg_to_cv2(msg)                           
        self.current_frame = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
        results = self.score_frame(self.current_frame)
        self.current_frame = self.plot_boxes(results, self.current_frame)
        cv2.imshow('frame', self.current_frame)

        end_time = time()
        fps = 1/np.round(end_time - start_time, 3)
        # print(f"Frames Per Second : {fps}")
        cv2.waitKey(1) 

    def Depth_Image_listener_callback(self, msg:Image):
        self.depth_frame = self.br.imgmsg_to_cv2(msg)                           

    def load_model(self):
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model

    def score_frame(self, frame):
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1].cpu().numpy(), results.xyxyn[0][:, :-1].cpu().numpy()
        return labels, cord

    def class_to_label(self, x):
        return self.classes[int(x)]

    def plot_boxes(self, results, frame):
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.2:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
                x_mid = (x2-x1)/2.0 + x1
                y_mid = (y2-y1)/2.0 + y1
                color = self.color_detection(frame,round(x_mid), round(y_mid))

                if self.class_to_label(labels[i]) == 'bottle':
                    z_mid = np.median(self.depth_frame[y1:y2,x1:x2])
                    # z_mid = 0
                    print(self.class_to_label(labels[i]),x_mid, y_mid, z_mid, color)

        return frame
    
    # Run with Webcam 
    # def __call__(self):
    #     self.player = self.get_video()
    #     while True:
    #         start_time = time()
    #         ret, frame = self.player.read()
    #         assert ret
    #         results = self.score_frame(frame)
    #         frame = self.plot_boxes(results, frame)
    #         cv2.imshow('frame', frame)

    #         end_time = time()
    #         fps = 1/np.round(end_time - start_time, 3)
    #         print(f"Frames Per Second : {fps}")
    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             break

    def color_detection(self,frame,x,y):

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        pixel_center = hsv_frame[y,x]

        H_value = pixel_center[0] * 360.0 / 179.0 # convert range from 0-179 to 0-360
        S_value = pixel_center[1] * 100.0 / 255.0 # convert range from 0-255 to 0-100
        V_value = pixel_center[2] * 100.0 / 255.0 # convert range from 0-255 to 0-100

        if V_value <= 30:
            color = 'black'
        elif S_value <= 20.0:
            if V_value <= 60:
                color = 'gray'
            else:
                color = 'white'
        else:    
            if H_value <= 15.0:
                color = 'red'
            elif H_value <= 35.0:
                color = 'orange'
            elif H_value <= 65.0:
                color = 'yellow'
            elif H_value <= 160.0:
                color = 'green'
            elif H_value <= 205.0:
                color = 'cyan'
            elif H_value <= 280.0:
                color = 'blue'
            elif H_value <= 300.0:
                color = 'purple'
            elif H_value <= 340.0:
                color = 'pink'
            else:
                color = 'red'
        return color
    
def main(args=None):
    rclpy.init(args=args)
    controller = ObjectDetection()
    # controller()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()

if __name__=='__main__':
    main()
