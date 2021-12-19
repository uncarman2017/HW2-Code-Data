import cv2
import os
import shutil


def clean_data():
    cwd = os.getcwd()
    del_file(cwd + "/Data/train/")

    for i in range(1, 10):
        raw_dir = "./Raw_Data/train_Raw_Data/"

        for filename in os.listdir(raw_dir + str(i)):
            if filename.startswith(str(i) + "-") and filename.endswith(".png") and filename.index("p") == 9:
                sub_dir = raw_dir + str(i) + "/"
                image_cv = cv2.imread(sub_dir + filename)
                cv2.imshow("image", image_cv)
                if image_cv.shape[0] == 50 and image_cv.shape[1] == 50:
                    src = os.path.join(cwd + "/Raw_Data/train_Raw_Data/" + str(i) + "/", filename)
                    dst = os.path.join(cwd + "/Data/train/", filename)
                    shutil.copy(src, dst)
        print(i)
        print("process finish!")


def del_file(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            del_file(c_path)
        else:
            os.remove(c_path)

if __name__ == "__main__":
    clean_data()
