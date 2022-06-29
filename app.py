import torch
import cv2
import threading
import time
import yaml
import argparse

class camThread(threading.Thread):
    def __init__(self, cam_cfg, cfg, yolo_model):
        threading.Thread.__init__(self)
        self.previewName = cam_cfg['name']
        self.camID = cam_cfg['id']
        self.cfg = cfg
        self.yolo_model = yolo_model
    def run(self):
        print ('Starting ' + self.previewName)
        camPreview(self.previewName, self.camID, self.cfg, self.yolo_model)

def camPreview(previewName, camID, cfg, yolo_model):
    p_time = 0
    c_time = 0

    yolo_cfg = cfg['yolo_detector']
    plot_cfg = cfg['plots']

    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.namedWindow(previewName)
    # cap = cv2.VideoCapture(camID + cv2.CAP_DSHOW)
    cap = cv2.VideoCapture(camID)
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            # detect person by yolo
            cam_res = cfg['camera_res']
            frame = cv2.resize(frame, (cam_res[0], cam_res[1]))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = yolo_model(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            results = results.xyxy[0].cpu().detach().numpy().tolist()

            # draw bbox
            for x1, y1, x2, y2, score, _ in results:
                if score < yolo_cfg['conf_score_thres']:
                    continue
                x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), plot_cfg['bbox_color'], plot_cfg['thickness'])
            
            # show people count
            frame = cv2.putText(
                frame, 
                f'Person count: {len(results)}',
                plot_cfg['ppl_count_text_pos'],
                font,
                plot_cfg['font_scale'],
                plot_cfg['ppl_count_text_color'],
                plot_cfg['thickness'],
                cv2.LINE_AA)
            
            # show fps
            c_time = time.time()
            fps = 1/(c_time-p_time)
            p_time = c_time
            frame = cv2.putText(
                frame, 
                f'FPS:{str(int(fps))}',
                plot_cfg['fps_text_pos'],
                font,
                plot_cfg['font_scale'],
                plot_cfg['fps_text_color'],
                plot_cfg['thickness'],
                cv2.LINE_AA)

            cv2.imshow(previewName, frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
    cv2.destroyWindow(previewName)

def main():
    # load config file
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', '--config', default='configs/config.yaml',
                        help='path to config file')
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.cfg, encoding='utf-8'))

    # load yolo model
    yolo_cfg = cfg['yolo_detector']
    device = torch.device(yolo_cfg['device'])
    yolo_model = torch.hub.load(
        yolo_cfg['yolo_path'],
        'custom',
        source='local',
        path=yolo_cfg['yolo_weight']).to(device)
    yolo_model.classes = [0]  # person

    # turn on cam
    cam_cfgs = cfg['cameras']
    for cam_cfg in cam_cfgs:
        thread = camThread(cam_cfg, cfg, yolo_model)
        thread.start()

if __name__ == '__main__':
    main()