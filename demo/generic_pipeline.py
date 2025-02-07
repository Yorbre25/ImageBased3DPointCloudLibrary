import cv2
from pointcloudlib import *


image_path1 = "IMG_2630.JPEG"
image_path2 = "IMG_2631.JPEG"

image1 = cv2.imread(image_path1, cv2.IMREAD_COLOR)
image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

image2 = cv2.imread(image_path2, cv2.IMREAD_COLOR)
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)


depth_estimation1 = depth_estimation_from_image(image1)
depth_estimation2 = depth_estimation_from_image(image2)

source = point_cloud_from_image(image1, depth_estimation1)
target = point_cloud_from_image(image2, depth_estimation2)

cl, ind = source.remove_statistical_outlier(nb_neighbors=20, std_ratio=2)
source = source.select_by_index(ind)
cl, ind = target.remove_statistical_outlier(nb_neighbors=20, std_ratio=2)
target = target.select_by_index(ind)

result = align_point_clouds(source, target)
print(vars(result))
draw_two_clouds(source, target, result.transformation, diff_color=False)
