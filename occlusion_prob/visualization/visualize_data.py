# coding=utf-8
from drawing_package.drawing_package.data_interface.data_utils import add_time_offset
from drawing_package.drawing_package.draw_function import draw_data
from drawing_package.drawing_package.data_interface.load_data import *
from drawing_package.drawing_package.processing.preprocessing import extract_position_dimension_data, remove_border_objects_from_data, \
    center_data, correct_extents, correct_extents_data
from drawing_package.drawing_package.globals.colors import *
from drawing_package.drawing_package.globals.config import *

folder = '2021-06-29-13-49-20'
dir = 'C:/Users/cyrin/{}/'.format(folder)
calibration_folder = '2021-06-29'

click = True
plot_HD_map_lanes = False
save_images = False

#--------------------------------------------------------------------------------------------
views = ['s50_s_cam_near'] #, 's50_s_cam_far' , 's40_n_cam_near', 's40_n_cam_far']

cams = ['s50_s_cam_near'] #, 's50_s_cam_far' , 's40_n_cam_near', 's40_n_cam_far']

local_frame = 'road'

image_dirs = []
for view in views:
    image_dirs.append(dir + '{}_images'.format(view))

save_dir = dir + 'saved_images'

config.update({'click': click, 'save_images': save_images, 'save_dir': save_dir, 'local_frame': local_frame,
               'plot_HD_map_lanes': plot_HD_map_lanes, 'sample_lanes': True,
               'old_projection': False, 'calibration_folder': calibration_folder})

colors_cam = [lightBlue, blue, orange, red]
colors_radar = [green, pink, purple]


if __name__ == '__main__':
    drawer_data = []

    for cam, color in zip(cams, colors_cam):
        measurement = load_measurements(dir + '{}_detections/'.format(cam))
        data = {'type': 'bbox', 'data': measurement, 'color_mode': 'sensor', 'sensor_name': cam, 'views': [cam]}
        drawer_data.append(data)

    views.append('top_view')

    draw_data(views, drawer_data, image_dirs, config)

