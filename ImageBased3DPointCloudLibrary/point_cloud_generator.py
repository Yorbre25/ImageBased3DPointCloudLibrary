import numpy as np
import open3d as o3d
import cv2

def point_cloud_from_image(image: np.ndarray, 
                           depth_estimation: np.ndarray,
                           scale_ratio:int = 100,
                           camera_intrinsics=None):
    pcd_raw = make_pcd(image, depth_estimation, scale_ratio, camera_intrinsics)
    return pcd_raw


def make_pcd(image, depth_estimation, scale_ratio=100, camera_intrinsics=None):
    height, width = depth_estimation.shape 

    # Resize image to fit depth_estimation
    image = cv2.resize(image, (width, height)) 

    # Depth estimation should not have a zero value
    depth_estimation = np.maximum(depth_estimation, 1e-5) 
    depth_estimation = scale_ratio / depth_estimation
    x,y,z = pixel_to_point(depth_estimation, camera_intrinsics)
    points = np.stack((x,y,z), axis=-1)

    cloud = o3d.geometry.PointCloud()
    mask = points[:,:,2] < 1e2 #Remove points with small z

    cloud.points = o3d.utility.Vector3dVector(points[mask].reshape(-1,3))
    # open3d uses a 1 dimentional array for coloring the points. Its value is from 0 to 1.
    cloud.colors = o3d.utility.Vector3dVector(image[mask].reshape(-1,3)/255)

    return cloud

def get_intrinsics(H,W,fov=55):
    f = 0.5 * W / np.tan(0.5 * fov * np.pi / 180)
    cx = 0.5 * W
    cy = 0.5 * H
    return np.array([[f, 0, cx], 
                     [0, f, cy], 
                     [0, 0, 1]])



def pixel_to_point(depth_estimation, camera_intrinsics):

    height, width = depth_estimation.shape
    if camera_intrinsics is None:
        camera_intrinsics = get_intrinsics(height, width, fov=55)
    
    fx,fy = camera_intrinsics[0,0], camera_intrinsics[1,1]
    cx,cy = camera_intrinsics[0,2], camera_intrinsics[1,2]
    
    x = np.linspace(0, width-1, width)
    y = np.linspace(0, height-1, height)
    u, v = np.meshgrid(x, y)

    x_over_z = (u - cx) / fx
    y_over_z = (v - cy) / fy

    z = depth_estimation / np.sqrt(1 + x_over_z**2 + y_over_z**2)
    x = x_over_z * z
    y = y_over_z * z

    return x,y,z


