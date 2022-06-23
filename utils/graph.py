import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st


def draw_graph(data):
    data = sorted(data)
    min90, max90, min95, max95 = data_90(data)
    ax = sns.kdeplot(data)
    print(ax.lines)
    x = ax.lines[0].get_xdata()
    y = ax.lines[0].get_ydata()
    
    four_min_y = list(x[y>=0.4])[0]
    four_max_y = list(x[y>=0.4])[-1]
    
    third_min_y = list(x[y>=0.35])[0]
    third_max_y = list(x[y>=0.35])[-1]
    
    # plt.axhline(0.4, color = 'r', linestyle = '--', label = 'density = 0.4')
    # plt.axhline(0.35, color = 'b', linestyle = '--', label = 'density = 0.35')
    
    # plt.axvline(min90, color = 'g', linestyle = '--', label = 'Left,Right 20%')
    # plt.axvline(max90, color = 'g', linestyle = '--', label = 'Left,Right 20%')
    
    # # plt.axvline(third_min_y, color = 'b', linestyle = '--')
    # # plt.axvline(third_max_y, color = 'b', linestyle = '--')
    
    # plt.axvline(min95, color = 'purple', linestyle = '--', label = 'Left,Right 10%')
    # plt.axvline(max95, color = 'purple', linestyle = '--', label = 'Left,Right 10%')
    
    # plt.legend()
    plt.show()
    print(round(four_min_y, 3), round(four_max_y, 3))
    # list_x = list(x[y>=0.4])
    # print(list_x)

def data_90(data):
    lt_90 = int(len(data)) * (40/100) / 2
    rt_90 = int(len(data)) * (40/100) / 2
    min90 = data[int(lt_90)]
    max90 = data[-int(rt_90)]
    
    lt_95 = int(len(data)) * (20/100) / 2
    rt_95 = int(len(data)) * (20/100) / 2
    min95 = data[int(lt_95)]
    max95 = data[-int(rt_95)]
    
    print(min90, max90)
    
    return min90, max90, min95, max95


def multi_occ(box_list):
    iou_list = []
    bbox_area_list = bbox_area(box_list)
    main_bbox = max(bbox_area_list)
    idx = bbox_area_list.index(main_bbox)
    
    for i in box_list:
        iou = IoU(i, box_list[idx])
        iou_list.append(iou)
        
    iou_list.remove(1)
    return iou_list
    

def IoU(box_1, box_2):
    # box = (x1, y1, x2, y2)
    box_1_area = (box_1[2] - box_1[0] + 1) * (box_1[3] - box_1[1] + 1)
    box_2_area = (box_2[2] - box_2[0] + 1) * (box_2[3] - box_2[1] + 1)

    x1 = max(box_1[0], box_2[0])
    y1 = max(box_1[1], box_2[1])
    x2 = min(box_1[2], box_2[2])
    y2 = min(box_1[3], box_2[3])

    w = max(0, x2 - x1 + 1)
    h = max(0, y2 - y1 + 1)

    inter = w * h
    iou = inter / (box_1_area + box_2_area - inter)
    return iou


def over_nms(box_1, box_2):
    # box = (x1, y1, x2, y2)
    box_1_area = (box_1[2] - box_1[0] + 1) * (box_1[3] - box_1[1] + 1)
    # box_2_area = (box_2[2] - box_2[0] + 1) * (box_2[3] - box_2[1] + 1)

    x1 = max(box_1[0], box_2[0])
    y1 = max(box_1[1], box_2[1])
    x2 = min(box_1[2], box_2[2])
    y2 = min(box_1[3], box_2[3])

    w = max(0, x2 - x1 )
    h = max(0, y2 - y1 )

    inter = w * h
    pad_inter = box_1_area - inter
    Over_nms = pad_inter / box_1_area
    return Over_nms


def bbox_area(box_list):
    box_area_list = []
    for x1, y1, x2, y2 in box_list:
        box_area = (x2 - x1) * (y2 - y1)
        box_area_list.append(box_area)
    return box_area_list