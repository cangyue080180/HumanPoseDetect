import socket
import os
import struct
import threading
import cv2
import numpy
import datetime


def get_time_now():
    return datetime.datetime.now().strftime('%m-%d %H:%M:%S')


class TcpClient:
    # 每个摄像头处理线程都独享一个tcp连接
    def __init__(self, server_ip, server_port, camera_id, room_id):
        self.tcp_server_ip = server_ip
        self.tcp_server_port = server_port
        self.camera_id = camera_id
        self.room_id = room_id
        # 压缩参数，后面cv2.imencode将会用到，对于jpeg来说，15代表图像质量，越高代表图像质量越好为 0-100，默认95
        self.encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 30]

        self.is_stop = True
        self.is_room_video_send = False
        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_socket.connect((self.tcp_server_ip, self.tcp_server_port))
        print(f'{get_time_now()} connection tcp_server success')

    def reconnection(self):
        print(f'{get_time_now()} start reconnection')
        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_socket.connect((self.tcp_server_ip, self.tcp_server_port))
        print(f'{get_time_now()} reconnection tcp_server success')
        self.start()

    def __socket_receive(self, data_len):
        result = self.tcp_socket.recv(data_len)
        receive_len = len(result)
        while receive_len < data_len:
            temp_result = self.tcp_socket.recv(data_len - receive_len)
            receive_len += len(temp_result)
            result += temp_result
        return result

    def send_img(self, img):
        # TODO:可考虑是否需要另起一个线程进行发送来提高性能
        if self.is_room_video_send and not self.is_stop:
            result, imgencode = cv2.imencode('.jpg', img, self.encode_param)
            data = numpy.array(imgencode)
            send_file = data.tobytes()

            # 发送图像数据包头
            file_size = len(send_file)
            packet_header = struct.pack('<BII', 2, file_size + 4, self.camera_id)

            if self.is_room_video_send:
                try:
                    self.tcp_socket.send(packet_header)
                    self.tcp_socket.send(send_file)
                    # print(f'send {file_size}')
                except OSError as e:
                    print(f'send_img_with_exception: {str(e)}')

    def start(self):
        self.is_stop = False
        # 发送角色数据包
        packet_role = struct.pack('<BIB', 3, 1, 2)
        self.tcp_socket.send(packet_role)

        # 开始等待接收控制命令
        t = threading.Thread(target=self.__receive, args=())
        t.daemon = True
        t.start()

    def stop(self):
        self.is_stop = True

    def __receive(self):
        print(f'tcp_recv_thread: {threading.currentThread().name}')
        # 等待接收图像传输命令
        is_except = False
        while not self.is_stop:
            try:
                packet_video_request_header = self.__socket_receive(5)
                packet_video_request_type, packet_video_request_len = struct.unpack('<BI', packet_video_request_header)
                if packet_video_request_type == 1:  # 图像传输控制命令包
                    room_id, video_status = struct.unpack('<IB', self.__socket_receive(5))
                    if self.room_id == room_id:
                        if video_status == 1:  # 发送此房间图像
                            self.is_room_video_send = True
                        else:  # 关闭此房间图像
                            self.is_room_video_send = False
                        print(f'{get_time_now()} is_video_send: {self.is_room_video_send}')
            except ConnectionError:
                is_except = True
                break
        # 尝试重连
        if is_except:
            self.is_room_video_send = False
            try:
                self.tcp_socket.close()
            except ConnectionError:
                print(f'{get_time_now()} close last socket with exception')
            self.reconnection()
        else:
            self.tcp_socket.close()
