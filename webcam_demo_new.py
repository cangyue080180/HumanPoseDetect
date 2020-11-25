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
import datetime
from pPose_nms import write_json

from align import AlignPoints
import threading
import tcpClient
import sched

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
    '''
    读取数据库中该用户的数据记录赋值为初始值
    :param aged: PoseInfo对象实例
    :return: None
    '''
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


def tensor_to_value(tensor_item):
    if isinstance(tensor_item,torch.Tensor):
        tensor_item = tensor_item.item()
    return tensor_item


def pose_detect_with_video(aged_id, classidx,human_box, parse_pose_demo_instance):
    use_aged = ages[aged_id]
    classidx = int(classidx)

    # detect if a new day come
    now_date = time.strftime('%Y-%m-%dT00:00:00', time.localtime())
    if not now_date == use_aged.date:  # a new day
        aged_status_reset(use_aged)
        parse_pose_demo_instance.is_first_frame = True
        use_aged.date = now_date

    if parse_pose_demo_instance.is_first_frame:  # 第一帧，开始计时
        # 从服务器获取当天的状态记录信息，进行本地值的更新，防止状态计时被重置
        aged_status_sync(use_aged)
        parse_pose_demo_instance.is_first_frame = False
    else:
        last_pose_time = time.time() - parse_pose_demo_instance.last_time  # 上一个状态至今的时间差，单位为s
        if use_aged.status == PoseStatus.Sit.value:
            use_aged.timesit += last_pose_time
        elif use_aged.status == PoseStatus.Down.value:
            use_aged.timedown += last_pose_time
        elif use_aged.status == PoseStatus.Lie.value:
            use_aged.timelie += last_pose_time
        elif use_aged.status == PoseStatus.Stand.value:
            use_aged.timestand += last_pose_time
        else:
            use_aged.timeother += last_pose_time

    parse_pose_demo_instance.last_time = time.time()

    if classidx == 0:
        now_status = PoseStatus.Sit.value
    elif classidx == 1:
        now_status = PoseStatus.Lie.value
    elif classidx == 2:
        now_status = PoseStatus.Stand.value
    elif classidx == 3:
        now_status = PoseStatus.Down.value
    else:
        now_status = PoseStatus.Other.value

    now_date_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    if not now_status == use_aged.status:  # 新的行为发生
        if not now_date_time == use_aged.datetime:  # 根据实际情况，每秒只记录一次状态更改操作
            temp_detail_pose_info = DetailPoseInfo(agesInfoId=aged_id, dateTime=now_date_time, status=now_status)
            # 写数据记录到数据库表DetaiPoseInfo
            detail_pose_url = Conf.Urls.DetailPoseInfoUrl
            http_result = HttpHelper.create_item(detail_pose_url, temp_detail_pose_info)
    use_aged.datetime = now_date_time

    is_outer_chuang=False
    #  因为床的矩形坐标是在原图压缩1/2之后的值，所以下面的值也需要压缩1/2
    xmin,ymin,xmax,ymax=int(human_box[0]/2),int(human_box[1]/2),int(human_box[2]/2),int(human_box[3]/2)
    if xmin>Conf.bed_max_x or ymin>Conf.bed_max_y or xmax<Conf.bed_min_x or ymax<Conf.bed_min_y:
        is_outer_chuang=True

    use_aged.isalarm = False
    # 判断当前状态是否需求报警
    if is_outer_chuang:  # TODO：这里的给值是不对的，需要赋予识别服务的对应的需要报警的状态值
        if now_status == PoseStatus.Down.value or now_status == PoseStatus.Lie.value:
            use_aged.isalarm = True

    use_aged.status = now_status

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
                                                         date=time.strftime('%Y-%m-%dT00:00:00', time.localtime()), dateTime=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                                                         timeStand=0, timeSit=0, timeLie=0, timeDown=0, timeOther=0)
                            if len(result['result']) > 0:
                                # 更新被监护对象各种状态的时间值
                                # TODO: 未检测到人的时候也应该更新数据库，只不过状态为NONE
                                human_first = result['result'][0]
                                pose_detect_with_video(aged.id, float(human_first['class']),human_first['bbox'], self)

                            break  # TODO：目前只考虑每个房间只有一个人的情况，所有目前一次就跳出了循环
            except KeyboardInterrupt:
                break

        while (writer.running()):
            pass
        writer.stop()


def write_database():
    """
    每1秒更新一次数据库表PoseInfo的记录信息
    :return:
    """
    pose_url = Conf.Urls.PoseInfoUrl + '/UpdateOrCreatePoseInfo'

    for aged in ages.values():
        temp_pose_info = PoseInfo(agesInfoId=aged.agesinfoid,
                                  date=aged.date,
                                  dateTime=aged.datetime,
                                  timeStand=int(float(aged.timestand)),
                                  timeSit=int(float(aged.timesit)),
                                  timeLie=int(float(aged.timelie)),
                                  timeDown=int(float(aged.timedown)),
                                  timeOther=int(float(aged.timeother)),
                                  isAlarm=aged.isalarm,
                                  status=aged.status
                                  )
        http_result = HttpHelper.create_item(pose_url, temp_pose_info)
    scheduler.enter(1, 0, write_database, ())


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

    # 定时调度来更新数据库
    scheduler = sched.scheduler(time.time, time.sleep)
    scheduler.enter(1, 0, write_database, ())

    for camera in current_server.cameraInfos:  # 遍历本服务器需要处理的摄像头
        tcpClient_instance = tcpClient.TcpClient('127.0.0.1', 8008, camera.id, camera.roomInfoId)
        tcpClient_instance.start()

        print('start new ParsePoseDemo')
        temp_task_instance = ParsePoseDemo(camera, 'examples/resttttt', 1, pose_model, pos_reg_model,
                                           tcpClient_instance, False)
        temp_task_instance.start()
        pose_parse_instance_list.append(temp_task_instance)

    scheduler.run()

    for instance in pose_parse_instance_list:
        instance.stop()



