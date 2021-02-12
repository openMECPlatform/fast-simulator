#!/usr/bin/python


class Config:
    # Service constraint

    MEC1 = "MEC1"
    MEC2 = "MEC2"
    MEC3 = "MEC3"
    MEC4 = "MEC4"

    MEC1_IP = "192.168.0.50"
    MEC2_IP = "192.168.0.51"
    MEC3_IP = "192.168.0.52"
    MEC4_Ip = "192.168.0.53"

    ETCD_PORT = "2379"
    CAR_PORT = "4567"
    MEC_PATH = ""
    # Define parameters for similator

    LATENCY_ARG = 0.001  # 1 ms corresponding to 1000 distance unit

    # 5s latency
    int_mec1 = 25.18
    int_mec2 = 22.87
    int_mec3 = 20.67
    int_mec4 = 21.32

    min_dist = 18

    pa_lat = 6

