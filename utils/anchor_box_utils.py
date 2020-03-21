import numpy as np

num_grids_ssd300 = [[38, 38],
                    [19, 19],
                    [10, 10],
                    [5, 5],
                    [3, 3],
                    [1, 1]]
grid_step_ssd300 = [[8, 8],
                    [16, 16],
                    [32, 32],
                    [64, 64],
                    [100, 100],
                    [300, 300]]
grid_size_ssd300 = [[30, 30],
                    [60, 60],
                    [111, 111],
                    [162, 162],
                    [213, 213],
                    [264, 264],
                    [315, 315]]
aspect_ratio_ssd300 = [[2,],
                       [2, 3,],
                       [2, 3,],
                       [2, 3,],
                       [2,],
                       [2,]]

def generate_default_box(input_size, num_grids, step, size, aspect_ratio):
    default_box = []
    for l in range(len(num_grids)):
        for x in range(num_grids[l][0]):
            for y in range(num_grids[l][1]):
                cx = (x + 0.5) * step[l][0]
                cy = (y + 0.5) * step[l][1]
                w = size[l][0]
                h = size[l][1]
                default_box.append([cx, cy, w, h])
                w = np.sqrt(size[l][0] * size[l+1][0])
                h = np.sqrt(size[l][1] * size[l+1][1])
                default_box.append([cx, cy, w, h])
                for ar in aspect_ratio[l]:
                    w = size[l][0] * np.sqrt(ar)
                    h = size[l][1] / np.sqrt(ar)
                    default_box.append([cx, cy, w, h])
                    w = size[l][0] / np.sqrt(ar)
                    h = size[l][1] * np.sqrt(ar)
                    default_box.append([cx, cy, w, h])
    default_box = np.array(default_box)
    default_box[:, :2] /= input_size
    default_box[:, 2:] /= input_size
    return default_box
                    


    