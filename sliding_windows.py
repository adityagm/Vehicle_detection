import numpy as np


def sliding_windows(image_size, init_size, x_overlap, y_step, x_range, y_range, scale):
    image_w, image_h = image_size[0], image_size[1]
    # print(image_w, image_h)
    # init_size = (256, 256)
    grid = []

    for y in range(int(y_range[0] * image_h), int(y_range[1] * image_h), int(y_step * image_h)):
        win_width = int(init_size[0] + (scale * (y - (y_range[0] * image_h))))
        win_height = int(init_size[1] + (scale * (y - (y_range[0] * image_h))))
        if y + win_height > int(y_range[1] * image_h) or win_width > image_w:
            break

        x_step = int((1 - x_overlap) * win_width)

        for x in range(int(x_range[0] * image_w), int(x_range[1] * image_w), x_step):
            if 0 <= x <= 1280 and 0 <= (x + win_width) <= 1280:
                grid.append((x, y, x + win_width, y + win_height))

    return grid


# def make_grid_1(image_size, init_size):
#     image_w, image_h = image_size[0], image_size[1]
#
#     grid = []
#     return(grid.append())


