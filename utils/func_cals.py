import cv2

from utils.graph import *


def CutOut(draw_img, img, boundingbox_list, pad, file_name):
    img_width = img.shape[1]
    img_height = img.shape[0]
    pad = pad
    
    cut_box = (0 + pad, 0 + pad, img_width - pad, img_height - pad)
    cut_over = over_nms(boundingbox_list[0], cut_box)
    
    if (0 == boundingbox_list[0][0]) or (img_width == boundingbox_list[0][2]):
        if cut_over < 0.3:
            img_cut = draw_img.copy()
            cv2.imwrite('output/test/cut_over/' + file_name, img_cut)
            print('---------Save image in Cut-Out!---------')
            

def Occlusion():
    