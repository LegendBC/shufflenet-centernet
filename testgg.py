import sys
CENTERNET_PATH = '/root/lbc/shufflenet-centernet/CenterNet/src/lib'
sys.path.insert(0, CENTERNET_PATH)

from detectors.detector_factory import detector_factory
from opts import opts

MODEL_PATH = '/root/lbc/shufflenet-centernet/CenterNet/exp/ctdet/model_last.pth'
TASK = 'ctdet' # or 'multi_pose' for human pose estimation
backbone = 'shufflenet'
opt = opts().init('{} --load_model {} --arch {}'.format(TASK, MODEL_PATH, backbone).split(' '))
detector = detector_factory[opt.task](opt)

img = '/root/lbc/shufflenet-centernet/000001.jpg'
ret = detector.run(img)['results']
print(ret)