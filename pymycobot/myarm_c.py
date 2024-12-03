# 读取myarm_c的角度并发送
import socket
from pymycobot import MyArmC, MyArmM
import serial.tools.list_ports
import time

def get_port(): # 获取所有串口号
    port_list = serial.tools.list_ports.comports()
    i = 1
    res = {}
    for port in port_list:
        print("{} - {}".format(i, port.device))
        res[str(i)] = port.device
        i += 1
    return res

def main():
    port_dict = get_port()
    port_c = input("input myArm C port: ")
    c_port = port_dict[port_c]
    c = MyArmC(c_port, debug=False)
    c = MyArmC('/dev/ttyACM1', debug=False)
    HOST = '127.0.0.1'
    PORT = 8001
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 请求连接
    client.connect((HOST, PORT))
    # TODO：在接受到信息后，可以同步打开视频的录制，从而保证两者的同步性，
    #       同时将信息的获取与录制频率同时设定在50hz，保证数据的一致性
    while True:
        angle = c.get_joints_angle()
        print(angle)
        if angle is not None:
            data = '\n' + str(angle)
            client.send(data.encode('utf-8'))

if __name__ == "__main__":
    main()
