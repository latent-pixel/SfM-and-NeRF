import numpy as np
import os


def getMatchesFromFile(file_path):
    file_data = []
    with open(file_path, "r") as file:
        for idx, line in enumerate(file.readlines()):
            if idx != 0:
                str_line = line.strip().split()[1:]
                row = []
                for i in range(len(str_line)):
                    if i in (0, 1, 2):
                        elem = np.int16(str_line[i])
                    else:
                        if i % 3 == 2:
                            elem = np.int16(str_line[i])
                        else:
                            elem = np.float32(str_line[i])
                    row.append(elem)
                file_data.append(row)
    return file_data


def createMatchesArray(folder_dir):
    all_matches = np.empty((0, 5, 2))
    rgb_vals = []
    for file in sorted(os.listdir(folder_dir)):
        if "matching" in file:
            curr_image = int(file.split('.')[0][-1])-1
            file_data = getMatchesFromFile(os.path.join(folder_dir, file))
            matches_array = np.zeros(shape=(len(file_data), 5, 2))
            for idx, row in enumerate(file_data):
                rgb_vals.append((row[0], row[1], row[2]))
                matches_array[idx, curr_image, 0] = row[3]           # points in the current image
                matches_array[idx, curr_image, 1] = row[4]
                row = row[5:]
                for i in range(len(row)):
                    if i%3 == 0:
                        image_no = row[i]
                        matches_array[idx, image_no-1, 0] = row[i+1]
                        matches_array[idx, image_no-1, 1] = row[i+2]
            all_matches = np.append(all_matches, matches_array, axis=0)
    return all_matches, np.array(rgb_vals)

