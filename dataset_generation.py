import os
from PIL import Image
import random
import shutil


def resize_images(origin_path, destination_path, width, height):
    if len(os.listdir(destination_path)) != 0:      # guard clause to catch whether there are any files in destination path
        return

    for file in os.listdir(origin_path):
        os.chdir(origin_path)
        file_name, _ = os.path.splitext(file)
        image = Image.open(file)
        size, _ = image.size
        if size < 100: continue
        resized_image = image.resize((width, height))
        os.chdir(destination_path)
        resized_image.save(f"{file_name}.png")


def create_multiple_directories(path, names):
    for name in names:
        try:
            os.mkdir(path + "//" + name)
        except FileExistsError:
            pass  # directory already exists


def create_multilayer_directories(path, parent_names, child_names):
    create_multiple_directories(path, parent_names)
    os.chdir(path)
    for directory in os.listdir():
        create_multiple_directories(path + "//" + directory, child_names)


def copy_all_files(from_path, to_path):
    if get_total_files_no(to_path) > 0:
        print("There are existing files in destination path")
        return
    os.chdir(from_path)
    for file in os.listdir():
        os.chdir(from_path)
        image = Image.open(file)
        os.chdir(to_path)
        image.save(file)
    print("Copying finished")


def get_total_files_no(path):
    os.chdir(path)
    return len([f for f in os.listdir(path) if os.path.isfile(f)])


def move_random_number_of_files(from_path, to_path, total_files_no, r=1.0):
    os.chdir(from_path)
    files = os.listdir(from_path)
    if r == 1.0:
        for file in os.listdir():
            try:
                shutil.move(from_path + "//" + file, to_path + "//" + file)
            except Exception as e:
                print(e)
    else:
        for file in random.sample(files, int(r * total_files_no)):
            os.chdir(from_path)
            try:
                shutil.move(from_path + "//" + file, to_path + "//" + file)
            except Exception as e:
                print(e)
    print("Moving files finished")


# this function sorts benign data based on type
def prepare_benign_type_data(from_path, to_path):
    os.chdir(from_path)
    for file in os.listdir():
        os.chdir(from_path)
        file_extension = os.path.splitext(file)[0].split('_')[-1]
        image = Image.open(file)
        os.chdir(to_path + "//" + file_extension)  # e.g. "D://Data//BenignType//acm"
        image.save(file)
    print("Data preparation finished")


def prepare_malicious_class_data(from_path, to_path, csv):
    # df = pd.read_csv("C://Users//Kuba//Desktop//Images//trainLabels.csv", header=0)
    # print(df.dtypes)
    # print(df['Class'].unique())
    pass


def generate_dataset(paths):
    temp_path = "D://Temp"

    # guard clause if there are any files in any destination folder
    if (len(os.listdir(paths["train"])) != 0 or
            len(os.listdir(paths["test"])) != 0 or
            len(os.listdir(paths["valid"])) != 0):
        return

    if paths['dataset'] == "":
        if len(os.listdir(temp_path)) == 0:
            copy_all_files(paths["data"], temp_path)
            files_no = get_total_files_no(temp_path)
            move_random_number_of_files(temp_path, paths["train"], files_no, 0.7)
            move_random_number_of_files(temp_path, paths["test"], files_no, 0.2)
            move_random_number_of_files(temp_path, paths["valid"], files_no)
    else:
        for dir_name in os.listdir(paths["data"]):
            if len(os.listdir(temp_path)) == 0:
                copy_all_files(paths["data"] + "//" + dir_name, temp_path)
                files_no = get_total_files_no(temp_path)
                move_random_number_of_files(temp_path, paths["train"] + "//" + dir_name, files_no, 0.7)
                move_random_number_of_files(temp_path, paths["test"] + "//" + dir_name, files_no, 0.2)
                move_random_number_of_files(temp_path, paths["valid"] + "//" + dir_name, files_no)


directories = ['_Train', '_Test', '_Valid']
ben_types = ['acm', 'ax', 'cpl', 'dll', 'drv', 'efi', 'exe', 'mui', 'ocx', 'scr', 'sys', 'tsp']
mal_classes = ['Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5', 'Class_6', 'Class_7', 'Class_8', 'Class_9']
ben_paths = {
    "raw_img": "D://Data//BenignRaw",
    "data": "D://Data//Benign",
    "train": "D://Datasets//Benign//_Train",
    "test": "D://Datasets//Benign//_Test",
    "valid": "D://Datasets//Benign//_Valid",
    "dataset": ""
}
mal_paths = {
    "raw_img": "D://Data//MaliciousRaw",
    "data": "D://Data//Malicious",
    "train": "D://Datasets//Malicious//_Train",
    "test": "D://Datasets//Malicious//_Test",
    "valid": "D://Datasets//Malicious//_Valid",
    "dataset": ""
}
ben_type_paths = {
    "data": "D://Data//BenignType",
    "train": "D://Datasets//BenignType//_Train",
    "test": "D://Datasets//BenignType//_Test",
    "valid": "D://Datasets//BenignType//_Valid",
    "dataset": "D://Datasets//BenignType"
}
mal_class_paths = {
    "data": "D://Data//MaliciousClass",
    "train": "D://Datasets//MaliciousClass//_Train",
    "test": "D://Datasets//MaliciousClass//_Test",
    "valid": "D://Datasets//MaliciousClass//_Valid",
    "dataset": "D://Datasets//MaliciousClass"
}
ben_paths_299 = {
    "data": "D://Data//Benign_299",
    "train": "D://A//BenignMalicious_299//_Train//Benign",
    "test": "D://A//BenignMalicious_299//_Test//Benign",
    "valid": "D://A//BenignMalicious_299//_Valid//Benign",
    "dataset": ""
}
mal_paths_299 = {
    "data": "D://Data//Malicious_299",
    "train": "D://A//BenignMalicious_299//_Train//Malicious",
    "test": "D://A//BenignMalicious_299//_Test//Malicious",
    "valid": "D://A//BenignMalicious_299//_Valid//Malicious",
    "dataset": ""
}

resize_images(ben_paths["raw_img"], ben_paths["data"])
resize_images(mal_paths["raw_img"], mal_paths["data"])
generate_dataset(ben_paths)
generate_dataset(mal_paths)
generate_dataset(ben_type_paths)
generate_dataset(mal_class_paths)
generate_dataset(ben_paths_299)
generate_dataset(mal_paths_299)
# create_multilayer_directories(ben_type_paths["dataset"], directories, ben_types)
# create_multiple_directories(ben_type_destination, directories)
# os.chdir(ben_type_destination)
# for directory in os.listdir():
#     create_multiple_directories(ben_type_destination + "//" + directory, ben_types)

create_multilayer_directories(mal_class_paths["dataset"], directories, mal_classes)
# create_multiple_directories(mal_class_destination, directories)
# os.chdir(mal_class_destination)
# for directory in os.listdir():
#     create_multiple_directories(mal_class_destination + "//" + directory, mal_classes)
