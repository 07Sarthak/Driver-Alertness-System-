import shutil
from STMap_generator_lib.preprocess.preprocess_detect import pre_detect
from STMap_generator_lib.preprocess.preprocess_landmark import pre_landmark
from STMap_generator_lib.preprocess.preprocess_crop import pre_crop
from STMap_generator_lib.preprocess.preprocess_data import pre_data
from model_folder.testing_code import test

if __name__ == '__main__':
    pre_detect()
    pre_landmark()
    pre_crop()
    pre_data()
    map_path='/SAMPLE_DATA/cache/preprocess_data/video_5x5_ori.png'
    gt_path='/SAMPLE_DATA/ground_truth.txt'
    test(feature_map_path=map_path,ground_truth_path=gt_path)
    shutil.rmtree('/SAMPLE_DATA/cache/preprocess_data')
    shutil.rmtree('/SAMPLE_DATA/cache/preprocess_crop')
    shutil.rmtree('/SAMPLE_DATA/cache/preprocess_detect')
    shutil.rmtree('/SAMPLE_DATA/cache/preprocess_landmark')
