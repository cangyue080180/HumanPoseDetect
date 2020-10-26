import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms

import torch.nn as nn
import torch.utils.data
import numpy as np
from opt import opt

from dataloader_webcam_zd import WebcamLoader, DetectionLoader, DetectionProcessor, DataWriter, crop_from_dets, Mscoco
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
from threading import Thread
import tcpClient

args = opt
args.dataset = 'coco'


def loop():
    n = 0
    while True:
        yield n
        n += 1


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


def pose_detect_with_video(aged_id, classidx, parse_pose_demo):
    use_aged = ages[aged_id]
    # classidx = classidx.item()
    classidx = int(classidx)

    # detect if a new day come
    now_date = time.strftime('%Y-%m-%dT00:00:00', time.localtime())
    if int(now_date[8:10]) > int(use_aged.date[8:10]):  # a new day
        aged_status_reset(use_aged)
    use_aged.date = now_date

    if parse_pose_demo.is_first_frame:  # 第一帧，开始计时
        parse_pose_demo.is_first_frame = False
    else:
        last_pose_time = time.time() - parse_pose_demo.last_time  # 上一个状态至今的时间差，单位为s
        # TODO：需要添加新的一天从新计时的情况。当是新的一天时候，设置is_first_fram=True即可
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
    if use_aged.status == PoseStatus.Down.value:
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

    def start(self):
        # start a thread to read frames from the file video stream
        t = Thread(target=self.parse, args=())
        t.daemon = True
        t.start()
        return self

    def parse(self):
        print('start parse')
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)

        data_loader = WebcamLoader(self.camera_info.videoAddress).start()
        (fourcc, fps, frameSize) = data_loader.videoinfo()

        det_loader = DetectionLoader(data_loader, batchSize=self.detbatch).start()
        det_processor = DetectionProcessor(det_loader).start()

        aligner = AlignPoints()

        # Data writer
        # save_path = os.path.join(args.outputpath, 'AlphaPose_webcam' + webcam + '.avi')

        writer = DataWriter(self.tcp_client, self.save_video, self.output_path, cv2.VideoWriter_fourcc(*'XVID'),
                            fps, frameSize, pos_reg_model=pos_reg_model, aligner=aligner).start()

        # 统计时间使用
        runtime_profile = {
            'dt': [],
            'pt': [],
            'pn': []
        }

        batch_size = self.detbatch
        while True:
            try:
                start_time = getTime()
                with torch.no_grad():
                    (inps, orig_img, im_name, boxes, scores, pt1, pt2) = det_processor.read()
                    if boxes is None or boxes.nelement() == 0:
                        writer.save(None, None, None, None, None, orig_img, im_name.split('/')[-1])
                        continue

                    ckpt_time, det_time = getTime(start_time)
                    runtime_profile['dt'].append(det_time)
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
                    ckpt_time, pose_time = getTime(ckpt_time)

                    hm = hm.cpu().data
                    writer.save(boxes, scores, hm, pt1, pt2, orig_img, im_name.split('/')[-1])
                    while not writer.result_Q.empty():
                        result = writer.result_Q.get()
                        print('classidx:', result['result'][0]['class'])
                        # for re in range(len(result['result'])):
                        #    print('classidx:', result['result'][re]['class'])
                        # 遍历本摄像头所在房间的老人信息
                        # 目前只考虑房间只有一个人，所以只循环一遍
                        for aged in self.camera_info.roomInfo.agesInfos:
                            if not aged.id in ages.keys():
                                ages[aged.id] = PoseInfo(agesInfoId=aged.id,
                                                         date=time.strftime('%Y-%m-%dT00:00:00', time.localtime()),
                                                         timeStand=0, timeSit=0, timeLie=0, timeDown=0, timeOther=0)
                            if len(result['result']) > 0:
                                # 更新被监护对象各种状态的时间值
                                pose_detect_with_video(aged.id, float(result['result'][0]['class']), self)

                            break
                        # 创建或更新PoseInfo数据库记录
                        pose_url = Conf.Urls.PoseInfoUrl + '/UpdateOrCreatePoseInfo'
                        http_result = HttpHelper.create_item(pose_url, ages[aged.id])

                    ckpt_time, post_time = getTime(ckpt_time)
            except KeyboardInterrupt:
                break

        while (writer.running()):
            pass
        writer.stop()


ages = {}  # 老人字典
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
    pos_reg_model.load_state_dict(torch.load('.\\42_model.ckpt'))
    pos_reg_model.eval()

    # 拼接url，参考接口文档
    get_current_server_url = Conf.Urls.ServerInfoUrl + "/GetServerInfo?ip=" + local_ip
    print(f'get {get_current_server_url}')

    current_server = HttpHelper.get_items(get_current_server_url)
    # print(current_server)
    print(len(current_server.cameraInfos))
    for camera in current_server.cameraInfos:  # 遍历本服务器需要处理的摄像头
        tcpClient_instance = tcpClient.TcpClient('154.8.225.243', 8008, camera.id, camera.roomInfoId)
        tcpClient_instance.start()

        print('start new ParsePoseDemo')
        temp_task_instance = ParsePoseDemo(camera, 'examples/resttttt', 1, pose_model, pos_reg_model,
                                           tcpClient_instance, False)
        temp_task_instance.start()
        break

    while True:
        try:
            pass
            time.sleep(1)
        except KeyboardInterrupt:
            break
