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
    all_matches = [[[] for i in range(5)] for i in range(5)]
    for file in sorted(os.listdir(folder_dir)):
        if "matching" in file:
            curr_image = int(file.split('.')[0][-1])-1
            file_data = getMatchesFromFile(os.path.join(folder_dir, file))
            for idx, row in enumerate(file_data):
                curated_row = [row[0], row[1], row[2], row[3], row[4]]  # points in the current image
                row = row[5:]
                for i in range(len(row)):
                    matches = curated_row.copy()
                    if i%3 == 0:
                        image_no = row[i]-1
                        matches.append(row[i+1])
                        matches.append(row[i+2])
                        all_matches[curr_image][image_no].append(matches)
                        # print(curr_image, image_no, matches)     
    all_matches = np.array(all_matches, dtype=object)
    return all_matches
