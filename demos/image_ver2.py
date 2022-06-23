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
    
    cutout = []
    left = []
    right = []
    front = []
    back = []
    model_GCN = GCN(3, 4, torch.tensor(_A).float())
    model_GCN.load_state_dict(torch.load("GCN/exp/2022_06_14/BestModel.pth"))
    model_GCN.eval()
    
    for file in files:
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
            for pose in poses:
                # print(pose)

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
                    

        if kp_dets[0].count_nonzero() != 0:
            # print(kp_dets[0].count_nonzero())
            if args.kp_bbox:
                bboxes = scale_coords(img.shape[2:], kp_dets[0][:, :4], im0.shape[:2]).round().cpu().numpy()
                for i, x1, y1, x2, y2 in enumerate(bboxes):
                    cv2.rectangle(im0, (int(x1), int(y1)), (int(x2), int(y2)), args.color_kp, thickness=args.line_thick)
                    pose_score = poses[i].mean(axis = 0)[2]
                    cv2.rectangle(im0, (x1, y2 - 10), (x1+40, y2), (0, 0, 255), -1)
                    cv2.putText(im0, f"score : {round(pose_score, 2)}", (int(x1), int(y2) - 10), cv2.FONT_ITALIC, 1, (255, 0, 0), 2)


        filename = '{}'.format(osp.splitext(osp.split(file)[-1])[0])
        filename += '.png'
        
        width= img_width
        height = img_height
        pad_wid = max(int(width * 0.025), 1)
        pad_hei = max(int(height * 0.025), 1)
        nkeypoint_thr = 1
        
        num_keypoint = 0
        for pose in poses:
            for x, y, score in pose:
                #if (not ((0+pad)<=x<=(width - pad)))  or (not ((0+pad)<=y<=(height-pad))):
                if 0 + pad_wid > x or x > width - pad_wid or 0 + pad_hei > y or y > height - pad_hei:
                    num_keypoint += 1

                    if num_keypoint >= nkeypoint_thr : 
                        cutout +=[filename]
                        cv2.imwrite('output/test/CutOut/' + filename, im0)
                        break
                
                
            keypoints = pose.copy()
            keypoints = np.append(keypoints,(keypoints[5:6] + keypoints[6:7])/2, axis = 0)
            keypoints = torch.tensor(keypoints).float()
            pred = model_GCN(keypoints.unsqueeze(0))
            
            if pred.argmax() == 0:
                cv2.imwrite('output/test/front/' + filename, im0)
                front.append(filename)
            elif pred.argmax() == 1:
                cv2.imwrite('output/test/back/' + filename, im0)
                back.append(filename)
            elif pred.argmax() == 2:
                cv2.imwrite('output/test/left/' + filename, im0)
                left.append(filename)
            else:
                cv2.imwrite('output/test/right/' + filename, im0)
                right.append(filename)
              
        if person_dets[0].count_nonzero() == 0:
            pass
            
        elif bbox_num == 1:
            img_1 = im0.copy()                        
            if (int(pose[15,1]) > bboxes[0,3]) or (int(pose[16,1]) > bboxes[0,3]):
                cv2.imwrite('output/test/Occlusion/' + filename, img_1)
                # print('---------Save image in Occlusion!---------')
                
            elif (int(pose[5,1]) < bboxes[0,1]) or (int(pose[6,1]) < bboxes[0,1]):
                cv2.imwrite('output/test/Occlusion/' + filename, img_1)
                # print('---------Save image in Occlusion!---------')
            
            elif kp_num <= 8:
                cv2.imwrite('output/test/Occlusion/' + filename, img_1)
                # print('---------Save image in Occlusion!---------')
        
        else:
            img_2 = im0.copy()
            for i in range(len(bbox_list)):                                               
                if (int(poses[i][15,1]) > bboxes[i,3]) or (int(poses[i][16,1]) > bboxes[i,3]):
                    cv2.imwrite('output/test/Occlusion/' + filename, img_2)
                    # print('---------Save image in Occlusion!---------')
            
            iou_list = multi_occ(bbox_list)
            for el in iou_list:
                if el > 0.15:
                    cv2.imwrite('output/test/Occlusion/' + filename, img_2)
                    # print('---------Save image in Occlusion!---------')
                    
            cv2.imwrite('output/test/MultiPerson/' + filename, img_2)
            # print('---------Save image in MultiPerson!---------')
        end_patch = time.time()
        print("total time: ", round(end_patch - start_patch, 3))
        print('\n')
    print("total time: ", round(time.time() - start, 3))