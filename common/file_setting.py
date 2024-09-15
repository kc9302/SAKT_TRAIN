import glob
import os
import logging


def make_check_points(model_name=str, dataset_name=str, date_info=str):

    logging.debug("model_name :" + str(model_name))
    logging.debug("dataset_name :" + str(dataset_name))
    logging.debug("datetime :" + date_info.strftime("%y%m%d_%H_%M_%S"))

    # check_points 폴더 존재 여부
    if not os.path.isdir("check_points"):
        os.mkdir("check_points")
    # check_points/model 폴더 존재 여부
    check_points_model_path = str(os.path.join("check_points", str(model_name)))
    if not os.path.isdir(check_points_model_path):
        os.mkdir(check_points_model_path)
    # check_points/model/dataset 폴더 존재 여부
    check_points_model_data_path = os.path.join(check_points_model_path, str(dataset_name))
    logging.debug(check_points_model_data_path)
    if not os.path.isdir(check_points_model_data_path):
        os.mkdir(check_points_model_data_path)
    # # check_points/model/dataset/datetime 폴더 존재 여부
    # check_points_model_data_datetime_path = os.path.join(check_points_model_data_path,
    #                                                      date_info.strftime("%y%m%d_%H_%M_%S"))
    # if not os.path.isdir(check_points_model_data_datetime_path):
    #     os.mkdir(check_points_model_data_datetime_path)
    return check_points_model_data_path


# Operation flow sequence 15.
def find_datasets_path(file_name=str):

    join_dataset_name = "**/" + str(file_name) + "*"
    # 인자로 받은 패턴과 이름이 일치하는 모든 파일과 디렉토리의 리스트를 반환
    # recursive=True로 설정하고 "**"를 사용하면 모든 하위 디렉토리까지 탐색한다.
    if len(glob.glob(join_dataset_name, recursive=True)) > 1:
        dataset_path = glob.glob(join_dataset_name, recursive=True)[1]
    if len(glob.glob(join_dataset_name, recursive=True)) == 1:
        dataset_path = glob.glob(join_dataset_name, recursive=True)[0]
    last_index = dataset_path.rfind("/")    
    dataset_directory = dataset_path[:last_index+1]
    return dataset_directory, dataset_path
