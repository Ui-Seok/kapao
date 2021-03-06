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
from GCN.model import GCN, _A
import time

font_italic = "FONT_ITALIC"


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
    parser.add_argument('--conf-thres-kp-face', type=float, default=0.5)
    parser.add_argument('--iou-thres-kp-face', type=float, default=0.4)
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


    img_dir = args.img_path
    files = glob.glob(img_dir + '/*.*')
    
    fold_list = ['Occlusion', 'CutOut', 'MultiPerson', 'BackgroundDominant', 'left', 'right', 'back', 'front', 'WrongTarget']
    for fs in fold_list:
        shutil.rmtree(f'output/test/{fs}')
        os.mkdir(f'output/test/{fs}')
    
    for file in files:
        cutout, left, right, front, back, occ_list, multi_person = [], [], [], [], [], [], []
        start_patch = time.time()
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
            # print('person_dets:', person_dets)
            # print('person_dets[0]:', person_dets[0])
            if args.bbox:
                bboxes = scale_coords(img.shape[2:], person_dets[0][:, :4], im0.shape[:2]).round().cpu().numpy()
                bbox_num = bboxes.shape[0]
                # print(bboxes)
                for i, (x1, y1, x2, y2) in enumerate(bboxes):
                    cv2.rectangle(im0, (int(x1), int(y1)), (int(x2), int(y2)), [0, 255, 0], thickness=args.line_thick)
                

        _, poses, _, _, _ = post_process_batch(data, img, [], [[im0.shape[:2]]], person_dets, kp_dets)

        if args.pose:
            pose_list = []
            for i, pose in enumerate(poses):
                # print(pose)
                pose_score = poses[i].mean(axis = 0)[2]
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
                    
                    # ?????? 0.6??? ???????????? ????????? -> 0.38 -> 0.5 -> 0.55 -> 0.57
                    if (pose[1,2] > 0.3 or pose[2,2] > 0.3) and (left_side > right_side + 0.5): 
                        pose_list.append('left')
                    elif (pose[1,2] > 0.3 or pose[2,2] > 0.3) and (left_side + 0.5 < right_side): 
                        pose_list.append('right')
                    elif (pose[1,2] and pose[2,2]) <= 0.3:
                        pose_list.append('back')
                    else: 
                        pose_list.append('front')
                        
                print(pose_score)
                x1, y1, x2, y2 = bboxes[i]
                #cv2.rectangle(im0, (int(x1), int(y2 - 8)), (int(x1+60), int(y2)), (0, 255, 0), -1)
                cv2.putText(im0, f"{round(pose_score, 4)}", (int(x1), int(y2) + 5), cv2.FONT_ITALIC, .3, (255, 0, 0), 1, )
                

        if kp_dets[0].count_nonzero() != 0:
            # print(kp_dets[0].count_nonzero())
            if args.kp_bbox:
                bboxes = scale_coords(img.shape[2:], kp_dets[0][:, :4], im0.shape[:2]).round().cpu().numpy()
                for i, (x1, y1, x2, y2) in enumerate(bboxes):
                    cv2.rectangle(im0, (int(x1), int(y1)), (int(x2), int(y2)), args.color_kp, thickness=args.line_thick)
                    

        filename = '{}'.format(osp.splitext(osp.split(file)[-1])[0])
        filename += '.png'
        
        
        
        for pose_angle in pose_list:
            cv2.imwrite(f'output/test/{pose_angle}/' + filename, im0)
        
        
        pad_wid = max(int(img_width * 0.025), 1)
        pad_hei = max(int(img_height * 0.025), 1)
        nkeypoint_thr = 1
        
        num_keypoint = 0
        for pose in poses:
            for x, y, score in pose:
                if 0 + pad_wid > x or x > img_width - pad_wid or 0 + pad_hei > y or y > img_height - pad_hei:
                    num_keypoint += 1

                    if num_keypoint >= nkeypoint_thr : 
                        cutout +=[filename]
                        cv2.imwrite('output/test/CutOut/' + filename, im0)

              
        if person_dets[0].count_nonzero() == 0:
            cv2.imwrite('output/test/WrongTarget/' + filename, im0)
        
        # for one person    
        elif bbox_num == 1:                 
            if (int(pose[15,1]) > bboxes[0,3]) or (int(pose[16,1]) > bboxes[0,3]):
                cv2.imwrite('output/test/Occlusion/' + filename, im0)
                occ_list.append(filename)
                
            elif (int(pose[5,1]) < bboxes[0,1]) or (int(pose[6,1]) < bboxes[0,1]):
                cv2.imwrite('output/test/Occlusion/' + filename, im0)
                occ_list.append(filename)
            
            elif kp_num <= 4:
                cv2.imwrite('output/test/Occlusion/' + filename, im0)
                occ_list.append(filename)
        
        # for multi person
        else:
            num_person = 0
            for i in range(len(bboxes)):                                               
                if (int(poses[i][15,1]) > bboxes[i,3]) or (int(poses[i][16,1]) > bboxes[i,3]):
                    cv2.imwrite('output/test/Occlusion/' + filename, im0)
                    occ_list.append(filename)
                if poses[i].mean(axis = 0)[2] > 0.1:       
                    num_person +=1 
            
            iou_list = multi_occ(bboxes)
            for el in iou_list:
                if el > 0.15:
                    cv2.imwrite('output/test/Occlusion/' + filename, im0)
                    occ_list.append(filename)
            if num_person >= 2 :
                cv2.imwrite('output/test/MultiPerson/' + filename, im0)
                multi_person.append(filename)
        
        if len(set(occ_list+ multi_person + cutout)) >= 1 :
            pass
        else :
            if front:
                cv2.imwrite('output/test/Normal_front/' + filename, im0)
            elif back:
                cv2.imwrite('output/test/Normal_back/' + filename, im0)

        end_patch = time.time()
        print("patch time: ", round(end_patch - start_patch, 3))
        print('\n')
    print("total time: ", round(time.time() - start, 3))