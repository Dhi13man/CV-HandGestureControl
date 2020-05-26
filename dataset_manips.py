import os
import visualizer
from cv2 import imwrite
from string import ascii_letters, digits


def in_al_num(char):
    for i in ascii_letters:
        if char == ord(i):
            return True
    for i in digits:
        if char == ord(i):
            return True
    return False


def arrange_folder(name):
    location = os.path.join("Datasets", name)
    length = len([name for name in os.listdir(location)])
    print(length)
    for i in range(length):
        if not os.path.exists(location + "/%d.png" % (i + 1)):
            directory = os.listdir(location)
            maximum = directory[0].split('.')[0]
            for file_name in directory:
                if int(file_name.split('.')[0]) > int(maximum):
                    maximum = file_name.split('.')[0]
            os.rename(location + '/' + maximum + '.png', location + '/' + "%d.png" % (i + 1))


def arrange_all():
    arrange_folder("0")
    arrange_folder("hi5")
    arrange_folder("V")
    arrange_folder("midfin")
    arrange_folder("ok")
    arrange_folder("thumbsup")


def clear_dir(directory):
    i = 1
    while os.path.exists("Datasets/" + directory + "/%d.png" % i):
        os.remove("Datasets/" + directory + "/%d.png" % i)
        i += 1


def clear_datasets(dictionary=None):
    x = input("Are you sure? (Y/N): ")
    if x == 'Y':
        if dictionary is None:
            dictionary = {
                '0': '0',
                '1': 'hi5',
                '2': 'midfin',
                '3': 'ok',
                '4': 'thumbsup',
                '5': 'v'
            }
        for this_set in dictionary.values():
            if os.path.exists("Datasets/" + str(this_set)):
                clear_dir(this_set)
                os.removedirs("Datasets/" + str(this_set))


def image_save(gesture_obj, directory):
    i = 1
    while os.path.exists("Datasets/" + directory + "/%s.png" % i):
        i += 1
    f_name = "Datasets/" + directory + "/%s.png" % i
    imwrite(f_name, gesture_obj.area)


def other_keypress_check(directory_lis, gesture_obj):
    key = gesture_obj.key
    if not in_al_num(key) or key == ord('p') or key == ord('q'):
        return
    for this_set in directory_lis.values():
        if not os.path.exists("Datasets/" + str(this_set)):
            os.mkdir("Datasets/" + str(this_set))
    for this_button in directory_lis.keys():
        if key == ord(str(this_button)):
            print("%s stored" % str(directory_lis[this_button]))
            image_save(gesture_obj, str(directory_lis[this_button]))


def create_dataset(dictionary=None):
    G = visualizer.GestureDetector()
    if dictionary is None:
        dictionary = {
            '0': '0',
            '1': 'hi5',
            '2': 'midfin',
            '3': 'ok',
            '4': 'thumbsup',
            '5': 'v'
        }
    while G.showing:
        G.show_video_feed()
        other_keypress_check(dictionary, G)
    G.end_feed()


if __name__ == '__main__':
    pass
