import shutil
import sys
from pathlib import Path
FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[1].as_posix())  # add kapao/ to path

import glob
import torch
import argparse
import yaml
from utils.torch_utils import select_device
from utils.general import check_img_size, scale_coords
from utils.datasets import LoadImages
from models.experimental import attempt_load
from val import run_nms, post_process_batch
import cv2
import os
import os.path as osp
from utils.graph import *
import time


if __name__ == '__main__':
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--img-path', default='res/crowdpose_100024.jpg', help='path to image')

    # plotting options
    parser.add_argument('--bbox', action='store_true')
    parser.add_argument('--kp-bbox', action='store_true')
    parser.add_argument('--pose', action='store_true')
    parser.add_argument('--face', action='store_true')
    parser.add_argument('--color-pose', type=int, nargs='+', default=[255, 0, 255], help='pose object color')
    parser.add_argument('--color-kp', type=int, nargs='+', default=[0, 255, 255], help='keypoint object color')
    parser.add_argument('--line-thick', type=int, default=1, help='line thickness')
    parser.add_argument('--kp-size', type=int, default=1, help='keypoint circle size')
    parser.add_argument('--kp-thick', type=int, default=1, help='keypoint circle thickness')

    # model options
    parser.add_argument('--data', type=str, default='data/coco-kp.yaml')
    parser.add_argument('--imgsz', type=int, default=1280)
    parser.add_argument('--weights', default='kapao_l_coco.pt')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or cpu')
    parser.add_argument('--conf-thres', type=float, default=0.7, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--no-kp-dets', action='store_true', help='do not use keypoint objects')
    parser.add_argument('--conf-thres-kp', type=float, default=0.5)
    parser.add_argument('--conf-thres-kp-person', type=float, default=0.2)
    parser.add_argument('--iou-thres-kp', type=float, default=0.45)
    parser.add_argument('--overwrite-tol', type=int, default=25)
    parser.add_argument('--scales', type=float, nargs='+', default=[1])
    parser.add_argument('--flips', type=int, nargs='+', default=[-1])

    args = parser.parse_args()

    with open(args.data) as f:
        data = yaml.safe_load(f)  # load data dict

    # add inference settings to data dict
    data['imgsz'] = args.imgsz
    data['conf_thres'] = args.conf_thres
    data['iou_thres'] = args.iou_thres
    data['use_kp_dets'] = not args.no_kp_dets
    data['conf_thres_kp'] = args.conf_thres_kp
    data['iou_thres_kp'] = args.iou_thres_kp
    data['conf_thres_kp_person'] = args.conf_thres_kp_person
    data['overwrite_tol'] = args.overwrite_tol
    data['scales'] = args.scales
    data['flips'] = [None if f == -1 else f for f in args.flips]
    data['count_fused'] = False

    device = select_device(args.device, batch_size=1)
    print('Using device: {}'.format(device))

    model = attempt_load(args.weights, map_location=device)
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(args.imgsz, s=stride)  # check image size
    
    bbox_asp = []
    width_emp_list = []
    height_emp_list = []
    cut_over_list = []
    img_dir = args.img_path
    files = glob.glob(img_dir + '/*.*')
    
    fold_list = ['Occlusion', 'CutOut', 'MultiPerson', 'BackgroundDominant', 'left', 'right', 'back', 'front', 'WrongTarget']
    for fs in fold_list:
        shutil.rmtree(f'output/test/{fs}')
        os.mkdir(f'output/test/{fs}')
    
    for file in files:
        cv_img = cv2.imread(file)
        img_width = cv_img.shape[1]
        img_height = cv_img.shape[0]
        dataset = LoadImages(file, img_size=imgsz, stride=stride, auto=True)

        (_, img, im0, _) = next(iter(dataset))
        img = torch.from_numpy(img).to(device)
        img = img / 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

        out = model(img, augment=True, kp_flip=data['kp_flip'], scales=data['scales'], flips=data['flips'])[0]
        person_dets, kp_dets = run_nms(data, out)
        # print('person_dets:', person_dets[0].count_nonzero())
        # print('kp_dets:', kp_dets[0])
        
        if person_dets[0].count_nonzero() != 0:
            # print(person_dets[0][:,:4])
            if args.bbox:
                bbox_list = []
                bbox_height = []
                bbox_width = []
                bboxes = scale_coords(img.shape[2:], person_dets[0][:, :4], im0.shape[:2]).round().cpu().numpy()
                bbox_num = bboxes.shape[0]
                # print(bboxes)
                for x1, y1, x2, y2 in bboxes:
                    cv2.rectangle(im0, (int(x1), int(y1)), (int(x2), int(y2)), [0, 255, 0], thickness=args.line_thick)
                    bbox_list.append([x1, y1, x2, y2])
                    bbox_asp.append(abs((y2-y1)/(x2-x1)))
                    bbox_width.append(abs(x2 - x1))
                    bbox_height.append(abs(y2 - y1))
                

        _, poses, _, _, _ = post_process_batch(data, img, [], [[im0.shape[:2]]], person_dets, kp_dets)

        if args.pose:
            pose_list = []
            cnt_p = 0
            for pose in poses:
                rel = sum(pose[:, 2])
                rel_p = round((rel / 17), 4) * 100
                print(rel_p)
                if rel_p > 30:
                    if args.face:
                        for x, y, c in pose[data['kp_face']]:
                            cv2.circle(im0, (int(x), int(y)), args.kp_size, args.color_pose, args.kp_thick)
                    for seg in data['segments'].values():
                        pt1 = (int(pose[seg[0], 0]), int(pose[seg[0], 1]))
                        pt2 = (int(pose[seg[1], 0]), int(pose[seg[1], 1]))
                        cv2.line(im0, pt1, pt2, args.color_pose, args.line_thick)
                    if data['use_kp_dets']:
                        kp_num = 0
                        # print(pose[:5,2])       # face kp dets
                        for x, y, c in pose[5:,:]:
                            if c:
                                cv2.circle(im0, (int(x), int(y)), args.kp_size, args.color_kp, args.kp_thick)
                                kp_num += 1           
                        for x, y, c in pose[:5, :]:
                            if c:
                                cv2.circle(im0, (int(x), int(y)), args.kp_size, [255, 255, 0], args.kp_thick)
                        # print('Number of key points: ', kp_num)
                        
                        # pose_angle
                        left_side = pose[1,2] + pose[3,2]
                        right_side = pose[2,2] + pose[4,2]
                        # print(round(left_side, 4), round(right_side, 4))
                        
                        # 원래 0.6을 기준으로 하였음 -> 0.38 -> 0.5 -> 0.55 -> 0.57
                        if (pose[1,2] != 0 or pose[2,2] != 0) and (left_side > right_side + 0.565): 
                            pose_list.append('left')
                        elif (pose[1,2] != 0 or pose[2,2] != 0) and (left_side + 0.565 < right_side): 
                            pose_list.append('right')
                        elif (pose[1,2] and pose[2,2]) == 0:
                            pose_list.append('back')
                        else: 
                            pose_list.append('front')

                    

        if kp_dets[0].count_nonzero() != 0:
            # print(kp_dets[0].count_nonzero())
            if args.kp_bbox:
                bboxes = scale_coords(img.shape[2:], kp_dets[0][:, :4], im0.shape[:2]).round().cpu().numpy()
                for x1, y1, x2, y2 in bboxes:
                    cv2.rectangle(im0, (int(x1), int(y1)), (int(x2), int(y2)), args.color_kp, thickness=args.line_thick)

        filename = '{}'.format(osp.splitext(osp.split(file)[-1])[0])
        filename += '.png'
        
        # pad = 5
        
        pad_wid = max(int(img_width * 0.1), 1)
        pad_hei = max(int(img_height * 0.1), 1)
        
        pad_box = (0 + pad_wid, 0 + pad_hei, img_width - pad_wid, img_height - pad_hei)
        cut_over = over_nms(bbox_list[0], pad_box)
        cut_over_list.append(cut_over)
        # im1 = cv2.rectangle(im0, (pad_box[0], pad_box[1]), (pad_box[2], pad_box[3]), (0, 0, 255), thickness=1)
        # if cut_over < 0.80:
        #     cv2.imwrite('output/test/cut_test/' + filename, im1)
            
        # pose-angle 분류하는 코드 v3
        for pose_angle in pose_list:
            img_pose = im0.copy()
            cv2.imwrite(f'output/test/{pose_angle}/' + filename, img_pose)
            print(f'---------Save image in {pose_angle}!---------')
        
        # bbox가 0개면 WrongTarget, CutOut, Occlusion, Blur로 분류가능
        if person_dets[0].count_nonzero() == 0:
            img_0 = im0.copy()
            cv2.imwrite('output/test/WrongTarget/' + filename, img_0)
            print('---------Save image in WrongTarget!---------')
        
        # bbox가 1개면 imgsz <> boxsz 비교해서 background dominant로 분류 가능    
        elif bbox_num == 1:
            img_1 = im0.copy()
            em_width = max(bbox_list[0][0], cv_img.shape[1] - bbox_list[0][2])
            em_height = max(bbox_list[0][1], cv_img.shape[0] - bbox_list[0][3])
            # print((em_width / cv_img.shape[1]))
            # print((em_height / cv_img.shape[0]))
            # width_emp_list.append(em_width / cv_img.shape[1])
            # height_emp_list.append(em_height / cv_img.shape[0])
            # print(cv_img.shape) --> (H, W, C)
            # bbox밖에 skeleton이 찍힐경우 -> occlusion
            
            # Occlusion,,,,,,,,,,
            # for num in range(6, 17):
            #     for j in [0,1]:
            #         if (bboxes[0, j] > int(pose[num,j])) or (int(pose[num,j]) > bboxes[0, j+2]):
            #             # print('Occlusion Test Y line')
            #             cv2.imwrite('output/test/Occlusion/' + filename, img_1)
            #             print('---------Save image in Occlusion!---------')
            #         elif ((em_width / cv_img.shape[1]) >= 0.25) or ((em_height / cv_img.shape[0]) >= 0.29):
            #             cv2.imwrite('output/test/BackgroundDominant/' + filename, img_1)
            #             print('---------Save image in BackgorundDominant!---------')
                        
                    # if (bboxes[0, 0] > int(pose[num,0])) or (int(pose[num,0]) > bboxes[0, 2]):
                    #     print('Occlusion Test X line')
                    #     cv2.imwrite('output/test/Occlusion/' + filename, img_1)
                    
            if (int(pose[15,1]) > bboxes[0,3]) or (int(pose[16,1]) > bboxes[0,3]):
                cv2.imwrite('output/test/Occlusion/' + filename, img_1)
                print('---------Save image in Occlusion!---------')
                
            elif (int(pose[5,1]) < bboxes[0,1]) or (int(pose[6,1]) < bboxes[0,1]):
                cv2.imwrite('output/test/Occlusion/' + filename, img_1)
                print('---------Save image in Occlusion!---------')
            
            elif kp_num <= 8:
                cv2.imwrite('output/test/Occlusion/' + filename, img_1)
                print('---------Save image in Occlusion!---------')
            
            elif ((em_width / cv_img.shape[1]) >= 0.30) or ((em_height / cv_img.shape[0]) >= 0.25):
                cv2.imwrite('output/test/BackgroundDominant/' + filename, img_1)
                print('---------Save image in BackgorundDominant!---------')           
            
            if (0 == bbox_list[0][0]) or (cv_img.shape[1] == bbox_list[0][2]):
                # Duke의 경우: 3.19, 2.15  //  Market의 경우: 1.63, 2.52
                if (bbox_height[0] / bbox_width[0] >= 3.19) or (bbox_height[0] / bbox_width[0] <= 2.15):
                    img_cut = im0.copy()
                    cv2.imwrite('output/test/cut_test/' + filename, img_cut)
                    print('---------Save image in Cut-Out!---------')
                    
            if (0 == bbox_list[0][0]) or (cv_img.shape[1] == bbox_list[0][2]):
                if 0.35 < cut_over:
                    img_cut = im0.copy()
                    cv2.imwrite('output/test/CutOut/' + filename, img_cut)          
                    
            # if cut_over < 0.15:
            #     img_cut = im0.copy()
            #     cv2.imwrite('output/test/cut_over/' + filename, img_cut) 
                    
            # else:
                # BackgroundDominant 분류
                # if ((em_width / cv_img.shape[1]) >= 0.25) or ((em_height / cv_img.shape[0]) >= 0.29):
                #     cv2.imwrite('output/test/BackgroundDominant/' + filename, img_1)
                #     print('---------Save image in BackgorundDominant!---------')
                
                # bbox_list[0]: bbox의 좌표값들 [x1, y1, x2, y2]
                # if (0 == bbox_list[0][0]) or (cv_img.shape[1] == bbox_list[0][2]):
                # print('bbox 가로세로 비율:', bbox_height / bbox_width)
                # Duke의 경우: 3.19, 2.15  //  Market의 경우: 1.66, 2.56
                    # if (bbox_height[0] / bbox_width[0] >= 3.19) or (bbox_height[0] / bbox_width[0] <= 2.15):
                        # img_cut = im0.copy()
                        # cv2.imwrite('output/test/CutOut/' + filename, img_cut)
                        # print('---------Save image in Cut-Out!---------')
                #     else:
                #         cv2.imwrite('output/test/Normal/' + filename, img_1)
                #         print('---------Save image in Normal!---------')
                            
                # else:  
                #     cv2.imwrite('output/test/Normal/' + filename, img_1)
                #     print('---------Save image in Normal!---------')
        
        # bbox가 2개 이상이면 multiperson으로 분류 가능    
        else:
            img_2 = im0.copy()
            for i in range(len(bbox_list)):
                if (0 == bbox_list[i][0]) or (cv_img.shape[1] == bbox_list[i][2]):
                    if (bbox_height[i] / bbox_width[i] >= 3.19) or (bbox_height[i] / bbox_width[i] <= 2.15):
                        img_cut = im0.copy()
                        cv2.imwrite('output/test/cut_test/' + filename, img_cut)
                        print('---------Save image in Cut-Out!---------')
                        
                if (0 == bbox_list[i][0]) or (cv_img.shape[1] == bbox_list[i][2]):
                    if 0.35 < cut_over:
                        img_cut = im0.copy()
                        cv2.imwrite('output/test/CutOut/' + filename, img_cut)
                        
                # if cut_over < 0.15:
                #     img_cut = im0.copy()
                #     cv2.imwrite('output/test/cut_over/' + filename, img_cut)
                
                if (int(poses[i][15,1]) > bboxes[i,3]) or (int(poses[i][16,1]) > bboxes[i,3]):
                    cv2.imwrite('output/test/Occlusion/' + filename, img_2)
                    print('---------Save image in Occlusion!---------')
            
            # 사람끼리 가려진 경우
            iou_list = multi_occ(bbox_list)
            for el in iou_list:
                if el > 0.15:
                    cv2.imwrite('output/test/Occlusion/' + filename, img_2)
                    print('---------Save image in Occlusion!---------')
                
            cv2.imwrite('output/test/MultiPerson/' + filename, img_2)
            print('---------Save image in MultiPerson!---------')
            
    print("total time: ", time.time() - start)

        # cv2.imwrite('output/' + filename, im0)
    # print(bbox_asp)
    # draw_graph(bbox_asp)
    # draw_graph(width_emp_list)
    # draw_graph(height_emp_list)
    # print('가로: ', width_emp_list)
    # print('세로: ', height_emp_list)
    # print(cut_over_list)
    # draw_graph(cut_over_list)