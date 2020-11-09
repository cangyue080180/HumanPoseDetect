import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms

import torch.nn as nn
import torch.utils.data
import numpy as np
from opt import opt

from dataloader_webcam_new import WebcamLoader, DetectionLoader, DetectionProcessor, DataWriter, crop_from_dets, Mscoco
from yolo.darknet import Darknet
from yolo.util import write_results, dynamic_write_results
from SPPE.src.main_fast_inference import *

from SPPE.src.utils.img import im_to_torch
import os
import sys
from tqdm import tqdm
import time
from fn import getTime
import cv2
import clientdemo.Conf as Conf
from clientdemo.DataModel import *
import clientdemo.HttpHelper as HttpHelper
import time
from pPose_nms import write_json

from align import AlignPoints
import threading
import tcpClient

args = opt
args.dataset = 'coco'


class NeuralNet(nn.Module):
    def __init__(self, input_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 4)
        self.drop = nn.Dropout(p=0.2)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        return out


def aged_status_reset(aged):
    aged.timesit = 0
    aged.timelie = 0
    aged.timestand = 0
    aged.timedown = 0
    aged.timeother = 0
    aged.timein = None


def aged_status_sync(aged):
    # 拼接url，参考接口文档
    aged_today_status__url = Conf.Urls.PoseInfoUrl + "/" + str(aged.agesinfoid)
    print(f'get {aged_today_status__url}')

    try:
        aged_status_today = HttpHelper.get_items(aged_today_status__url)
    except Exception:  # 还没有数据记录
        return
    aged.timesit = aged_status_today.timeSit
    aged.timelie = aged_status_today.timeLie
    aged.timestand = aged_status_today.timeStand
    aged.timedown = aged_status_today.timeDown
    aged.timeother = aged_status_today.timeOther


def pose_detect_with_video(aged_id, classidx, parse_pose_demo):
    use_aged = ages[aged_id]
    classidx = int(classidx)

    # detect if a new day come
    now_date = time.strftime('%Y-%m-%dT00:00:00', time.localtime())
    if int(now_date[8:10]) > int(use_aged.date[8:10]):  # a new day
        aged_status_reset(use_aged)
        parse_pose_demo.is_first_frame = True
    use_aged.date = now_date

    if parse_pose_demo.is_first_frame:  # 第一帧，开始计时
        # 从服务器获取当天的状态记录信息，进行本地值的更新，防止状态计时被重置
        aged_status_sync(use_aged)
        parse_pose_demo.is_first_frame = False
    else:
        last_pose_time = time.time() - parse_pose_demo.last_time  # 上一个状态至今的时间差，单位为s
        if use_aged.status == PoseStatus.Sit.value:
            use_aged.timesit += last_pose_time
            use_aged.timesit = int(float(use_aged.timesit))
        elif use_aged.status == PoseStatus.Other.value:
            use_aged.timeother += last_pose_time
            use_aged.timeother = int(float(use_aged.timeother))
        elif use_aged.status == PoseStatus.Stand.value:
            use_aged.timestand += last_pose_time
            use_aged.timestand = int(float(use_aged.timestand))
        else:
            use_aged.timeother += last_pose_time
            use_aged.timeother = int(float(use_aged.timeother))

    parse_pose_demo.last_time = time.time()
    if classidx == 0:
        use_aged.status = PoseStatus.Sit.value
    elif classidx == 1:
        use_aged.status = PoseStatus.Other.value
    elif classidx == 2:
        use_aged.status = PoseStatus.Stand.value
    else:
        use_aged.status = PoseStatus.Other.value

    # 判断当前状态是否需求报警
    if use_aged.status == PoseStatus.Down.value:  # TODO：这里的给值是不对的，需要赋予识别服务的对应的需要报警的状态值
        use_aged.isalarm = True
    else:
        use_aged.isalarm = False


class ParsePoseDemo:
    def __init__(self, camera, out_video_path, detbatch, pose_model, pos_reg_model, tcp_client, save_video=False):
        self.camera_info = camera
        self.output_path = out_video_path
        self.detbatch = detbatch
        self.pose_model = pose_model
        self.pose_reg_model = pos_reg_model
        self.save_video = save_video
        self.is_first_frame = True
        self.last_time = 0
        self.tcp_client = tcp_client
        self.writer = None
        self.is_stop = True

    def start(self):
        # start a thread to read frames from the file video stream
        self.is_stop = False
        t = threading.Thread(target=self.parse, args=())
        t.daemon = True
        t.start()

    def stop(self):
        self.is_stop= True

    def parse(self):
        print(f'ParsePoseDemo_parse_thread: {threading.currentThread().name}')
        print('start parse')
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)

        data_loader = WebcamLoader(self.camera_info.videoAddress).start()
        (fourcc, fps, frameSize) = data_loader.videoinfo()

        det_loader = DetectionLoader(data_loader, batchSize=self.detbatch).start()
        det_processor = DetectionProcessor(det_loader).start()

        aligner = AlignPoints()

        writer = DataWriter(self.tcp_client, self.save_video, self.output_path, cv2.VideoWriter_fourcc(*'XVID'),
                            fps, frameSize, pos_reg_model=pos_reg_model, aligner=aligner).start()

        batch_size = self.detbatch
        while not self.is_stop:
            try:
                with torch.no_grad():
                    (inps, orig_img, im_name, boxes, scores, pt1, pt2) = det_processor.read()
                    if boxes is None or boxes.nelement() == 0:
                        writer.save(None, None, None, None, None, orig_img, im_name.split('/')[-1])
                        continue
                    # Pose Estimation

                    datalen = inps.size(0)
                    leftover = 0
                    if (datalen) % batch_size:
                        leftover = 1
                    num_batches = datalen // batch_size + leftover
                    hm = []
                    for j in range(num_batches):
                        inps_j = inps[j * batch_size:min((j + 1) * batch_size, datalen)].cuda()
                        hm_j = pose_model(inps_j)
                        hm.append(hm_j)
                    hm = torch.cat(hm)
                    hm = hm.cpu().data
                    writer.save(boxes, scores, hm, pt1, pt2, orig_img, im_name.split('/')[-1])
                    while not writer.result_Q.empty():
                        result = writer.result_Q.get()
                        # print('classidx:', result['result'][0]['class'])

                        for aged in self.camera_info.roomInfo.agesInfos:
                            if not aged.id in ages.keys():
                                ages[aged.id] = PoseInfo(agesInfoId=aged.id,
                                                         date=time.strftime('%Y-%m-%dT00:00:00', time.localtime()),
                                                         timeStand=0, timeSit=0, timeLie=0, timeDown=0, timeOther=0)
                            if len(result['result']) > 0:
                                # 更新被监护对象各种状态的时间值
                                # TODO: 未检测到人的时候也应该更新数据库，只不过状态为NONE
                                pose_detect_with_video(aged.id, float(result['result'][0]['class']), self)

                            break  # TODO：目前只考虑每个房间只有一个人的情况，所有目前一次就跳出了循环
                        # 创建或更新PoseInfo数据库记录
                        pose_url = Conf.Urls.PoseInfoUrl + '/UpdateOrCreatePoseInfo'
                        http_result = HttpHelper.create_item(pose_url, ages[aged.id])
                        # print(f'update_poseinfo_http_result: {http_result}')
            except KeyboardInterrupt:
                break

        while (writer.running()):
            pass
        writer.stop()


ages = {}  # 老人字典
pose_parse_instance_list = []
# 获取或设置本机IP地址信息
local_ip = '192.168.1.60'

if __name__ == "__main__":
    # Load pose model
    pose_dataset = Mscoco()
    if args.fast_inference:
        pose_model = InferenNet_fast(4 * 1 + 1, pose_dataset)
    else:
        pose_model = InferenNet(4 * 1 + 1, pose_dataset)
    pose_model.cuda()
    pose_model.eval()
    pos_reg_model = NeuralNet(17 * 3 * 9).cuda()
    pos_reg_model.load_state_dict(torch.load('./42_model.ckpt'))
    pos_reg_model.eval()

    # 拼接url，参考接口文档
    get_current_server_url = Conf.Urls.ServerInfoUrl + "/GetServerInfo?ip=" + local_ip
    print(f'get {get_current_server_url}')

    current_server = HttpHelper.get_items(get_current_server_url)
    # print(current_server)
    print(f'current_server.camera_count: {len(current_server.cameraInfos)}')

    for camera in current_server.cameraInfos:  # 遍历本服务器需要处理的摄像头
        tcpClient_instance = tcpClient.TcpClient('127.0.0.1', 8008, camera.id, camera.roomInfoId)
        tcpClient_instance.start()

        print('start new ParsePoseDemo')
        temp_task_instance = ParsePoseDemo(camera, 'examples/resttttt', 1, pose_model, pos_reg_model,
                                           tcpClient_instance, False)
        temp_task_instance.start()
        pose_parse_instance_list.append(temp_task_instance)

    while True:
        try:
            time.sleep(1)
        except KeyboardInterrupt:
            break

    for instance in pose_parse_instance_list:
        instance.stop()



