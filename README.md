### Count number of people using yolov5

#### Install
Clone this repo and yolov5 repo and install [requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) in a
[**Python>=3.7.0**](https://www.python.org/) environment, including
[**PyTorch>=1.7**](https://pytorch.org/get-started/locally/).

```bash
git clone https://github.com/quangtien19999/people_count_by_yolov5 #clone this repo
cd people_count_by_yolov5
git clone https://github.com/ultralytics/yolov5  # clone yolov5
cd yolov5
pip install -r requirements.txt  # install
```

#### Run
To run with demo video:
```bash
cd ..
python app.py
```

To run with webcams:
Edit [config_cam.yaml](https://github.com/quangtien19999/people_count_by_yolov5/blob/main/configs/config_cam.yaml) to fit your camera setting.
Then run the app with argument
```
python app.py --cfg configs/config_cam.yaml
