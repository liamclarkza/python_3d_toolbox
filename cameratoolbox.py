import camera
import plotting
from utils import create_dlc_points_2d_file
import logging
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
import pickle
import os

def save_cameras(project_dir, cameras, path=None):
    if path==None:
        path = os.path.join(os.path.expanduser(project_dir), "calibration/cameras")
    for cam in cameras:
        with open(os.path.join(path, f"{cam.name}.pickle"), 'wb') as file:
            pickle.dump(cam, file)
    print(f"Saved cameras to {path}")

def load_cameras(project_dir, path=None):
    if path==None:
        path = os.path.join(os.path.expanduser(project_dir), "calibration/cameras")
    camera_filepaths = [os.path.join(path, cam) for cam in sorted(os.listdir(path))]
    cameras = []
    for cam_path in camera_filepaths:
        with open(cam_path, 'rb') as file:
            cameras.append(pickle.load(file))
    return cameras

def save_points_2d_df(project_dir, points_2d_df, path=None):
    if path==None:
        path = os.path.join(os.path.expanduser(project_dir), "calibration/calib_points_2d_df.pickle")
    points_2d_df.to_pickle(path)
    print(f"Saved points to {path}")

def load_points_2d_df(project_dir, path=None):
    if path==None:
        path = os.path.join(os.path.expanduser(project_dir), "calibration/calib_points_2d_df.pickle")
    points_2d_df = pd.read_pickle(path)
    return points_2d_df


def save_calib_board(project_dir, calib_board, path=None):
    if path==None:
        path = os.path.join(os.path.expanduser(project_dir), "calibration/calib_board.pickle")
    with open(path, 'wb') as file:
        pickle.dump(calib_board, file)
    print(f"Saved calibration board to {path}")
    

def load_calib_board(project_dir, path=None):
    if path==None:
        path = os.path.join(os.path.expanduser(project_dir), "calibration/calib_board.pickle")
    with open(path, 'rb') as file:
        calib_board = pickle.load(file)
    return calib_board


def create_new_project(project_dir):
    root_path = os.path.expanduser(project_dir)
    exists = os.path.isdir(root_path)
    if exists:
        if input(f"Directory '{root_path}' already exists, do you want to overwrite? y/[n]: ") != 'y':
            print("Project creation aborted...")
            return
    os.makedirs(root_path, exist_ok=True)
    os.makedirs(os.path.join(root_path,'calibration/videos'), exist_ok=True)
    os.makedirs(os.path.join(root_path,'calibration/cameras'), exist_ok=True)
    print(f"Created project at '{root_path}'")
    return root_path


def find_corners(project_dir, calib_board_shape, calib_board_square_edge_len, frame_indices=None, input_video_filepaths=None, output_points_2d_df_filepath=None):
    """Finds calibration board corners in the calibration videos located in the <project_dir>/calibration/videos folder.
    Please ensure that the videos are synchronised with one another and are named in ascending order from camera 1 to n.
    This function also saves the calibration board settings and the found points to files in the <project_dir>/calibration folder.

    :param string project_dir: Path to your project folder
    :param (int, int) calib_board_shape: The numbers of corners along the x and y axis of the calibration board
    :param float calib_board_square_edge_len: The length in metres of a single square edge on the calibration board
    :param frame_indicies list[int]: If you want to only use a
    """
    if input_video_filepaths==None:
        video_path = os.path.join(os.path.expanduser(project_dir), "calibration/videos")
        input_video_filepaths = [os.path.join(video_path, vid) for vid in sorted(os.listdir(video_path))]
    #Create calibration board and points dataframe
    calib_board = camera.CalibrationBoard(calib_board_shape, calib_board_square_edge_len)
    save_calib_board(project_dir, calib_board)
    points_2d_df = camera.create_calib_points_2d(input_video_filepaths, calib_board, frame_indices=frame_indices)
    save_points_2d_df(project_dir, points_2d_df, output_points_2d_df_filepath)




def calibrate_intrinsics(project_dir, camera_model, camera_resolution, points_2d_df_filepath=None, calib_board_filepath=None):
    """Calibrates the intrinsic parameters for the cameras. It is assumed that all cameras are of the same type.

    :param string project_dir: Path to your project folder
    :param string camera_model: either 'standard' or 'fisheye'
    :param (int, int) camera_resolution: The resolution of the camera
    """
    points_2d_df = load_points_2d_df(project_dir, points_2d_df_filepath)
    calib_board = load_calib_board(project_dir, calib_board_filepath)
    cameras = []
    print("Calibrating camera intrinsics...")
    for i in range(points_2d_df['camera'].max()+1):
        cameras.append(camera.Camera(f"cam{i}", camera_model, camera_resolution))
    camera.calibrate_intrinsics_multi(points_2d_df, cameras, calib_board)
    print("Done.")
    save_cameras(project_dir, cameras)



def calibrate_extrinsics(project_dir, output_video_filepath=None):
    cameras = load_cameras(project_dir)
    points_2d_df = load_points_2d_df(project_dir)
    calib_board = load_calib_board(project_dir)
    print("Calibrating camera extrinsics...")
    rms = camera.calib_pairwise_extrinsics(points_2d_df, cameras, calib_board)
    print(f"Initial pairwise RMS reprojection error: {rms}")
    print("Creating initial 3D estimates for optimisation...")
    pairwise_points_3d_df = camera.get_pairwise_3d_points(points_2d_df, cameras)
    print("Performing sparse bundle adjustment...")
    points_3d_df = camera.run_calib_bundle_adjustment(pairwise_points_3d_df, points_2d_df, cameras)
    print("Done.")
    save_cameras(project_dir, cameras)
    if output_video_filepath:
        output_video_filepath = os.path.expanduser(output_video_filepath)
        print("Creating video...")
        plotting.create_animation(points_3d_df, output_video_filepath, cameras)
        print(f"Done. Video saved to {output_video_filepath}")


def dlc_to_3d(project_dir, dlc_filepaths, output_3d_point_df_filepath=None, output_video_filepath=None):
    cameras = load_cameras(project_dir)
    assert len(cameras) == len(dlc_filepaths), "Need a DLC file for each camera"
    points_2d_df = create_dlc_points_2d_file(dlc_filepaths)
    print("Creating initial 3D estimates for optimisation...")
    # points_2d_df = interpolate_df(points_2d_df, 0.5)
    # points_2d_df = points_2d_df[~points_2d_df['x'].isnull()]
    pairwise_points_3d_df = camera.get_pairwise_3d_points(points_2d_df[points_2d_df['likelihood']>0.5], cameras)
    print("Performing sparse bundle adjustment...")
    points_3d_df = camera.run_point_bundle_adjustment(pairwise_points_3d_df, points_2d_df, cameras)
    print("Done!")
    if output_3d_point_df_filepath:
        output_3d_point_df_filepath = os.path.expanduser(output_3d_point_df_filepath)
        print("Saving 3D points...")
        if str(output_3d_point_df_filepath).endswith('.csv'):
            points_3d_df.to_csv(output_3d_point_df_filepath)
        elif str(output_3d_point_df_filepath).endswith('.pickle'):
            points_3d_df.to_pickle(output_3d_point_df_filepath)
        elif str(output_3d_point_df_filepath).endswith('.h5'):
            points_3d_df.to_hdf(output_3d_point_df_filepath)
        else:
            print("Unsupported export format")
        print(f"Done. 3D points saved to {output_3d_point_df_filepath}")
    if output_video_filepath:
        output_video_filepath = os.path.expanduser(output_video_filepath)
        print("Creating video...")
        plotting.create_animation(points_3d_df, output_video_filepath, cameras)
        print(f"Done. Video saved to {output_video_filepath}")


