#!/bin/sh

#cd /root/AiForAged/video_transport_demo
nohup python -u webcam_demo_new.py > /var/log/HumanPose.log 2>&1 &
