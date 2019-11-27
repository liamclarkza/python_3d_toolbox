import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import logging
import pickle
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares
from typing import List
from progress_bar import print_progress_bar, create_multiproc_bar
import pandas as pd
import itertools
import time
import multiprocessing as mp

# ======================= CalibrationBoard Class =======================
class CalibrationBoard:
    shape = (0,0)
    square_edge_length = 0
    obj_points = []

    def __init__ (self, shape, square_edge_length):
        self.shape = shape
        self.square_edge_length = square_edge_length
        objp = np.zeros((1, shape[0]*shape[1], 3), np.float32)
        objp[0,:,:2] = np.mgrid[0:shape[0], 0:shape[1]].T.reshape(-1, 2)
        objp *= square_edge_length
        self.obj_points = objp

    def __str__(self):
        return str(self.obj_points)
# ===================== END CalibrationBoard Class =====================


# ============================ Camera Class ============================
class Camera:
    name = ''
    resolution = (0, 0)
    model = ''
    K = np.zeros((3, 3), dtype=np.float64) #Camera matrix
    D = np.zeros((4, 1), dtype=np.float64) #Distortion coefficients
    R = np.zeros((1, 1, 3), dtype=np.float64) #Rotation vector (Rodriguez rotation)
    T = np.zeros((1, 1, 3), dtype=np.float64) #Translation vector

    def __init__(self, name, model, resolution):
        self.name = name
        self.model = model
        self.resolution = resolution


    def calibrate_intrinsics(self, img_points, calib_board: CalibrationBoard):
        self.K, self.D, _ = calibrate_intrinsics(img_points, self.resolution, self.model, calib_board)


    def undistort_images(self, img_filepaths, output_dir):
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(self.K, self.D, np.eye(3), self.K, self.resolution, cv2.CV_16SC2)
        for fp in img_filepaths:
            logging.info(f"Undistorted: {fp}.")
            img = cv2.imread(fp)
            undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            filename = os.path.basename(fp)
            output_filepath = os.path.join(output_dir, f"{self.name}-{self.model}-undistorted-{filename}")
            cv2.imwrite(output_filepath, undistorted_img)

    

    def undistort_and_plot_image(self, img_filepath):
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(self.K, self.D, np.eye(3), self.K, self.resolution, cv2.CV_16SC2)
        img = cv2.imread(img_filepath)
        undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        plt.imshow(cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2RGB))
        plt.show()


    def get_camera_params(self):
        if self.model == 'fisheye':
        #FISHEYE: [R, R, R, T, T, T, fx, fy, cx, cy, alpha, k1, k2, k3, k4]
            params = np.empty(15)       
            params[0:3] = self.R.ravel()     #R
            params[3:6] = self.T.ravel()     #T
            params[6] = self.K[0,0]          #fx
            params[7] = self.K[1,1]          #fy
            params[8] = self.K[0,2]          #cx
            params[9] = self.K[1,2]          #cy
            params[10] = self.K[0,1]         #alpha
            params[11:15] = self.D.ravel()   #k1, k2, k3, k4
        elif self.model == 'standard':
        #STANDARD: [R, R, R, T, T, T, fx, fy, cx, cy, k1, k2, p1, p2, k3]
            params = np.empty(15)
            params[0:3] = self.R.ravel()   #R
            params[3:6] = self.T.ravel()   #T
            params[6] = self.K[0,0]        #fx
            params[7] = self.K[1,1]        #fy
            params[8] = self.K[0,2]        #cx
            params[9] = self.K[1,2]        #cy
            params[10:15] = self.D.ravel() #k1, k2, p1, p2, k3
        return params


    def set_from_camera_params(self, params):
        if self.model == 'fisheye':
        #FISHEYE: [R, R, R, T, T, T, fx, fy, cx, cy, alpha, k1, k2, k3, k4]
            self.K = np.zeros((3,3))     
            self.R = np.array(params[0:3]).reshape((3,1))
            self.T = np.array(params[3:6]).reshape((3,1))
            self.K[0,0] = params[6] #fx
            self.K[1,1] = params[7] #fy
            self.K[0,2] = params[8] #cx
            self.K[1,2] = params[9] #cy
            self.K[0,1] = params[10] #alpha
            self.D = np.array(params[11:15]).reshape((4,1))  #k1, k2, k3, k4
        elif self.model == 'standard':
        #STANDARD: [R, R, R, T, T, T, fx, fy, cx, cy, k1, k2, p1, p2, k3]
            self.K = np.zeros((3,3))     
            self.R = np.array(params[0:3]).reshape((3,1))
            self.T = np.array(params[3:6]).reshape((3,1))
            self.K[0,0] = params[6] #fx
            self.K[1,1] = params[7] #fy
            self.K[0,2] = params[8] #cx
            self.K[1,2] = params[9] #cy
            self.D = np.array(params[10:15]).reshape((5,1))  #k1, k2, p1, p2, k3
# ========================== END Camera Class ==========================


# =========================== Error Classes ===========================
class VideoReadError(Exception):
    pass

class ImageSizeError(Exception):
    pass

class TooFewImagesError(Exception):
    pass

class InvalidFrameError(Exception):
    pass

class InvalidCameraModelError(Exception):
    pass
# ========================= END Error Classes =========================


def create_calib_df(vid_filepath, calib_board: CalibrationBoard, frame_indices=None):
    cap = cv2.VideoCapture(vid_filepath)
    #check video opened successfully
    if not cap.isOpened():
        raise VideoReadError(f"Could not open video: {vid_filepath}")
    #determine which frames to look for checkerboard in
    video_frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    if frame_indices is None:
        frame_indices = range(int(video_frame_count))
    #check video has enough frames if frame_indices specified
    last_frame_index = max(frame_indices)
    if video_frame_count < last_frame_index:
        raise InvalidFrameError(f"Invalid frame indices. Video: {vid_filepath} only has {video_frame_count} frames. Asked for {last_frame_index}")
    #find calibration board in frames
    n_frames = len(frame_indices)
    print(f"Searching {n_frames} frames for {vid_filepath}")
    n_found = 0
    x = []
    y = []
    frames = []
    point_indices = []
    progress_bar_count = 0
    subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    print_progress_bar(progress_bar_count, n_frames, prefix='Progress:', suffix=f"Complete (calibration board found in {n_found} frames)", length=50)
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            # Find the chess board corners
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(
                gray, 
                calib_board.shape, 
                cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE
            )
            if ret:
                # Refine found corner positions
                n_found += 1
                corners = cv2.cornerSubPix(gray, corners ,(3,3),(-1,-1), subpix_criteria)
                for corner_idx, corner in enumerate(corners):
                    x.append(corner[0][0])
                    y.append(corner[0][1])
                    frames.append(frame_idx)
                    point_indices.append(corner_idx)
        progress_bar_count += 1
        print_progress_bar(progress_bar_count, n_frames, prefix='Progress:',
                           suffix=f"Complete (calibration board found in {n_found}/{progress_bar_count} frames)", length=50)

    data = {
        'x': x,
        'y': y,
        'frame': frames,
        'label': point_indices
    }
    df = pd.DataFrame(data)
    return df


def create_calib_df_par(vid_filepath, calib_board: CalibrationBoard, frame_indices=None):
    cap = cv2.VideoCapture(vid_filepath)
    #check video opened successfully
    if not cap.isOpened():
        raise VideoReadError(f"Could not open video: {vid_filepath}")
    #determine which frames to look for checkerboard in
    video_frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    if frame_indices is None:
        frame_indices = range(int(video_frame_count))
    #check video has enough frames if frame_indices specified
    last_frame_index = max(frame_indices)
    if video_frame_count < last_frame_index:
        raise InvalidFrameError(f"Invalid frame indices. Video: {vid_filepath} only has {video_frame_count} frames. Asked for {last_frame_index}")
    #find calibration board in frames
    n_frames = len(frame_indices)
    print(f"Searching {n_frames} frames for {vid_filepath}")

    n_workers = mp.cpu_count()
    q_in = mp.Queue(maxsize=n_workers)
    q_out = mp.Queue()
    #prepare progress bar
    inc, kill = create_multiproc_bar(n_frames)
    #prepare workers
    procs = []
    for w in range(n_workers):
        procs.append(mp.Process(target=worker, args=(q_in, q_out, inc, calib_board)))
    for p in procs:
        p.start()
    #start dispatch
    for frame_idx in frame_indices:
        logging.info(f"adding frame {frame_idx} to queue")
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            q_in.put((frame_idx, frame))
    for w in range(n_workers):
        q_in.put(None)
    #read results
    x = []
    y = []
    frames = []
    point_indices = []
    completed = 0
    while completed<n_workers:
        val = q_out.get()
        if val is None:
            completed += 1
        else:
            (frame_idx, corners) = val
            logging.info(f"reading frame {frame_idx} from queue")
            for corner_idx, corner in enumerate(corners):
                x.append(corner[0][0])
                y.append(corner[0][1])
                frames.append(frame_idx)
                point_indices.append(corner_idx)
    kill() #destroy progress bar process
    data = {
        'x': x,
        'y': y,
        'frame': frames,
        'label': point_indices
    }
    df = pd.DataFrame(data).sort_values(by=['frame', 'label'])
    return df


def worker(q_in, q_out, inc, calib_board):
    subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    while True:
        val = q_in.get()
        if val is None:
            q_out.put(None)
            break
        (frame_idx, frame) = val
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(
            gray, 
            calib_board.shape, 
            cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        if ret:
            # Refine found corner positions
            corners = cv2.cornerSubPix(gray, corners ,(3,3),(-1,-1), subpix_criteria)
            q_out.put((frame_idx, corners))
            inc(1)
        else:
            inc(0)


def create_calib_points_2d(vid_filepaths, calib_board: CalibrationBoard, frame_indices=None, output_filepath=None):
    n_videos = len(vid_filepaths)
    points_2d_df = pd.DataFrame()
    for i, v in enumerate(vid_filepaths):
        print(f"Processing video {i+1} of {n_videos}: {v}")
        df = create_calib_df_par(v, calib_board, frame_indices)
        df["camera"] = i
        points_2d_df = pd.concat((points_2d_df, df), ignore_index=True)
    return points_2d_df


def calibrate_intrinsics(img_points, resolution, model, calib_board: CalibrationBoard):
    #NOTE: img_points must be a numpy array with shape: (n_frames, n_calib_corners, 1, 2)
    #if you do not do this, the calibration will give a myterious type assertion error
    obj_points = np.array([calib_board.obj_points] * img_points.shape[0])
    frames_found = len(obj_points)
    if frames_found < 4:
        raise TooFewImagesError(f"Only found checkerboard in {frames_found} frames. Need at least 4 vaild frames to perform calibration.")
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(frames_found)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(frames_found)]
    term_criteria = cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6
    if model=='standard':
        calibration_flags = cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_TAUX_TAUY
        rms, _, _, _, _ = cv2.calibrateCamera(
            obj_points,
            img_points,
            resolution,
            K,
            D,
            rvecs,
            tvecs,
            calibration_flags,
            term_criteria
        )
    elif model=='fisheye':
        calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC #| cv2.fisheye.CALIB_CHECK_COND
        rms, _, _, _, _ = cv2.fisheye.calibrate(
            obj_points,
            img_points,
            resolution,
            K,
            D,
            rvecs,
            tvecs,
            calibration_flags,
            term_criteria
        )
    else:
        raise InvalidCameraModelError(f"Camera model not supported: {model}")
    logging.info(f"K={K.tolist()}")
    logging.info(f"D={D.tolist()}")
    logging.info(f"rms={rms}")
    return K, D, rms


def calibrate_intrinsics_multi(points_2d_df: pd.DataFrame, cameras: List[Camera], calib_board: CalibrationBoard):
    for cam_idx in range(points_2d_df['camera'].max()+1):
        img_points = np.array(points_2d_df.loc[points_2d_df['camera']==cam_idx, ['x', 'y']], dtype=np.float32)
        img_points = img_points.reshape((-1, calib_board.shape[0]*calib_board.shape[1], 1, 2))
        cameras[cam_idx].calibrate_intrinsics(img_points, calib_board)


def calib_pairwise_extrinsics(points_2d_df: pd.DataFrame, cameras: List[Camera], calib_board: CalibrationBoard):
    resolution = cameras[0].resolution
    model = cameras[0].model
    
    R_1 = np.array([[1, 0, 0],
                    [0, 0, -1],
                    [0, 1, 0]], dtype=np.float32)

    T_1 = np.array([[0, 0, 0]], dtype=np.float32).T

    R_rodrigues, _ = cv2.Rodrigues(R_1)
    cameras[0].R = R_rodrigues
    cameras[0].T = T_1
    rms_reprojection_errors = []
    for cam_idx in range(points_2d_df['camera'].max()):
        df_pair = points_2d_df[points_2d_df['camera']==cam_idx].merge(points_2d_df[points_2d_df['camera']==cam_idx+1], on=['frame', 'label'], suffixes=('_a','_b'))
        assert df_pair.shape[0] > 0, "No pairwise images"
        assert cameras[cam_idx+1].resolution == resolution
        assert cameras[cam_idx+1].model == model
        image_points_a = np.array(df_pair[['x_a', 'y_a']], dtype=np.float32).reshape((-1, calib_board.shape[0]*calib_board.shape[1], 1, 2))
        image_points_b = np.array(df_pair[['x_b', 'y_b']], dtype=np.float32).reshape((-1, calib_board.shape[0]*calib_board.shape[1], 1, 2))
        object_points = np.tile(calib_board.obj_points, (len(image_points_a), 1, 1, 1))
        
        # note retval is reprojection error
        flags = cv2.CALIB_FIX_INTRINSIC
        term = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
        if model == 'standard':
            rms, *_, R, T, _, _ = cv2.stereoCalibrate(object_points, image_points_a, image_points_b,
                                                        cameras[cam_idx].K, cameras[cam_idx].D,
                                                        cameras[cam_idx + 1].K, cameras[cam_idx + 1].D,
                                                        resolution, flags=flags, criteria=term)
        elif model == 'fisheye':
            #Note: fisheye stereoCalibrate has different requirements :/
            image_points_a = image_points_a.reshape((-1, 1, calib_board.shape[0]*calib_board.shape[1], 2))
            image_points_b = image_points_b.reshape((-1, 1, calib_board.shape[0]*calib_board.shape[1], 2))
            rms, *_, R, T = cv2.fisheye.stereoCalibrate(object_points, image_points_a, image_points_b,
                                                        cameras[cam_idx].K, cameras[cam_idx].D,
                                                        cameras[cam_idx + 1].K, cameras[cam_idx + 1].D,
                                                        resolution, flags=flags, criteria=term)
        else:
            raise InvalidCameraModelError(f"Camera model not supported: {model}")

        rms_reprojection_errors.append(rms)
        R_2 = R @ R_1
        T_2 = R @ T_1 + T
        R_rodrigues, _ = cv2.Rodrigues(R_2)
        cameras[cam_idx+1].R = R_rodrigues
        cameras[cam_idx+1].T = T_2
        R_1 = R_2
        T_1 = T_2
    return rms_reprojection_errors


def triangulate_points(cam_1: Camera, cam_1_points, cam_2: Camera, cam_2_points):
    assert cam_1_points.shape[0] > 0, "error: cam_1_points length is 0"
    assert cam_2_points.shape[0] > 0, "error: cam_2_points length is 0"
    if cam_1.model == 'standard':
        pts_1 = cv2.undistortPoints(cam_1_points, cam_1.K, cam_1.D)
    elif cam_1.model == 'fisheye':
        pts_1 = cv2.fisheye.undistortPoints(cam_1_points, cam_1.K, cam_1.D)
    else:
        raise InvalidCameraModelError(f"Camera model not supported: {cam_1.model}")
    R_mat_1, _ = cv2.Rodrigues(np.array(cam_1.R, dtype=np.float32))
    P1 = np.hstack((R_mat_1, cam_1.T))
    # 2nd cameras params
    if cam_2.model == 'standard':
        pts_2 = cv2.undistortPoints(cam_2_points, cam_2.K, cam_2.D)
    elif cam_2.model == 'fisheye':
        pts_2 = cv2.fisheye.undistortPoints(cam_2_points, cam_2.K, cam_2.D)
    else:
        raise InvalidCameraModelError(f"Camera model not supported: {cam_2.model}")
    R_mat_2, _ = cv2.Rodrigues(np.array(cam_2.R, dtype=np.float32))
    P2 = np.hstack((R_mat_2, cam_2.T))
    # triangulate using 2 cameras for initial guess points
    pts_4d = cv2.triangulatePoints(P1, P2, pts_1, pts_2)
    points_3d = (pts_4d[:3] / pts_4d[3]).T
    return points_3d


def get_pairwise_3d_points(points_2d_df, cameras: List[Camera]):
    n_cameras = len(cameras)
    camera_pairs = list(itertools.combinations(range(n_cameras), 2))
    df_pairs = pd.DataFrame(columns=['x','y','z'])
    #get pairwise estimates
    for (cam_a, cam_b) in camera_pairs:
        d0 = points_2d_df[points_2d_df['camera']==cam_a]
        d1 = points_2d_df[points_2d_df['camera']==cam_b]
        intersection_df = d0.merge(d1, how='inner', on=['frame','label'], suffixes=('_a', '_b'))
        if intersection_df.shape[0] > 0:
            logging.info(f"Found {intersection_df.shape[0]} pairwise points between camera {cam_a} and {cam_b}")
            cam_a_points = np.array(intersection_df[['x_a','y_a']], dtype=np.float).reshape((-1,1,2))
            cam_b_points = np.array(intersection_df[['x_b','y_b']], dtype=np.float).reshape((-1,1,2))
            points_3d = triangulate_points(cameras[cam_a], cam_a_points, cameras[cam_b], cam_b_points)
            intersection_df['x'] = points_3d[:, 0]
            intersection_df['y'] = points_3d[:, 1]
            intersection_df['z'] = points_3d[:, 2]
            df_pairs = pd.concat([df_pairs, intersection_df], ignore_index=True, join='outer', sort=False)
        else:
            logging.info(f"No pairwise points between camera {cam_a} and {cam_b}")

    points_3d_df = df_pairs[['frame', 'label', 'x','y','z']].groupby(['frame','label']).mean().reset_index()
    return points_3d_df


def rotate(points, rot_vecs):
    # takes numpy arrays of 3D points and corresponding Rodriguez rotation vectors
    # i.e. points[0] = [x,y,z] and camera_params[0] = [R,R,R]
    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid='ignore'):
        v = rot_vecs / theta
        v = np.nan_to_num(v)
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v

def project_standard(points, cam_params):
    #cam_params[0] = [R, R, R, T, T, T, fx, fy, cx, cy, k1, k2, p1, p2, k3]
    R = cam_params[:,:3]
    T = cam_params[:,3:6]
    fx = cam_params[:, 6]
    fy = cam_params[:, 7]
    cx = cam_params[:, 8]
    cy = cam_params[:, 9]
    k1 = cam_params[:, 10]
    k2 = cam_params[:, 11]
    p1 = cam_params[:, 12]
    p2 = cam_params[:, 13]
    k3 = cam_params[:, 14]

    points_proj = rotate(points, R)
    points_proj += T
    points_proj = points_proj[:, :2] / points_proj[:, 2, np.newaxis]
    
    radius_squared = np.sum(points_proj ** 2, axis=1)
    radial_dist = (1 + k1 * radius_squared + k2 * radius_squared ** 2 + k3 * radius_squared ** 3)

    points_proj[:, 0] = points_proj[:, 0] * radial_dist\
                        + 2 * p1 * points_proj[:, 0] * points_proj[:, 1]\
                        + p2 * (radius_squared + 2 * (points_proj[:, 0] ** 2))

    points_proj[:, 1] = points_proj[:, 1] * radial_dist\
                        + 2 * p2 * points_proj[:, 0] * points_proj[:, 1]\
                        + p1 * (radius_squared + 2 * (points_proj[:, 1] ** 2))

    points_proj[:, 0] = fx * points_proj[:, 0] + cx
    points_proj[:, 1] = fy * points_proj[:, 1] + cy
    return points_proj

def project_fisheye(world_pts, cam_params):
    #camera params: [R, R, R, T, T, T, fx, fy, cx, cy, alpha, k1, k2, k3, k4]
    R = cam_params[:,:3]
    T = cam_params[:,3:6]
    fx = cam_params[:,6]
    fy = cam_params[:,7]
    cx = cam_params[:,8]
    cy = cam_params[:,9]
    alpha = cam_params[:,10]
    k1 = cam_params[:,11]
    k2 = cam_params[:,12]
    k3 = cam_params[:,13]
    k4 = cam_params[:,14]
    #do the reprojection - https://docs.opencv.org/4.1.0/db/d58/group__calib3d__fisheye.html
    pts = rotate(world_pts, R)
    pts += T
    a = pts[:,0]/pts[:,2]
    b = pts[:,1]/pts[:,2]
    r = np.sqrt(a**2 + b**2)
    theta = np.arctan(r)
    theta_d = theta*(1 + k1*theta**2 + k2*theta**4 + k3*theta**6 + k4*theta**8)
    x = theta_d/r*a
    y = theta_d/r*b
    u = fx*(x+alpha*y) + cx
    v = fy*y + cy
    return np.dstack((u,v))


# ======================= Sparse Bundle Adjustment Sparsity Matrix =======================
def create_calib_sba_sparsity_matrix(n_cameras, n_points, camera_indices, point_indices):
    m = camera_indices.size * 2
    n = n_cameras * 15 + n_points * 3
    A = lil_matrix((m, n), dtype=int)
    i = np.arange(camera_indices.size)
    for s in range(15):
        A[2 * i, camera_indices * 15 + s] = 1
        A[2 * i + 1, camera_indices * 15 + s] = 1
    for s in range(3):
        A[2 * i, n_cameras * 15 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * 15 + point_indices * 3 + s] = 1
    return A

def create_point_sba_sparsity_matrix(n_points, camera_indices, point_indices):
    m = camera_indices.size * 2
    n = n_points * 3
    A = lil_matrix((m, n), dtype=int)
    i = np.arange(camera_indices.size)
    for s in range(3):
        A[2 * i, point_indices * 3 + s] = 1
        A[2 * i + 1, point_indices * 3 + s] = 1
    return A
# ===================== END Sparse Bundle Adjustment Sparsity Matrix =====================


# ======================= Sparse Bundle Adjustment Cost Functions =======================
def fisheye_point_cost_func(params, camera_params, n_cameras, n_points, camera_indices, point_indices, points_2d):
    camera_params = camera_params.reshape((n_cameras, 15))
    points_3d = params.reshape((n_points, 3))
    points_2d_reprojected = project_fisheye(points_3d[point_indices], camera_params[camera_indices])
    error = (points_2d_reprojected - points_2d).ravel()
    return error

def standard_point_cost_func(params, camera_params, n_cameras, n_points, camera_indices, point_indices, points_2d):
    camera_params = camera_params.reshape((n_cameras, 15))
    points_3d = params.reshape((n_points, 3))
    points_2d_reprojected = project_standard(points_3d[point_indices], camera_params[camera_indices])
    error = (points_2d_reprojected - points_2d).ravel()
    return error

def fisheye_calib_cost_func(params, n_cameras, n_points, camera_indices, point_indices, points_2d):
    camera_params = params[:n_cameras * 15].reshape((n_cameras, 15))
    points_3d = params[n_cameras * 15:].reshape((n_points, 3))
    points_2d_reprojected = project_fisheye(points_3d[point_indices], camera_params[camera_indices])
    error = (points_2d_reprojected - points_2d).ravel()
    return error

def standard_calib_cost_func(params, n_cameras, n_points, camera_indices, point_indices, points_2d):
    camera_params = params[:n_cameras * 15].reshape((n_cameras, 15))
    points_3d = params[n_cameras * 15:].reshape((n_points, 3))
    points_2d_reprojected = project_standard(points_3d[point_indices], camera_params[camera_indices])
    error = (points_2d_reprojected - points_2d).ravel()
    return error
# ===================== END Sparse Bundle Adjustment Cost Functions =====================


# ======================= Sparse Bundle Adjustment Routines =========================
def run_calib_bundle_adjustment(points_3d_df, points_2d_df, cameras: List[Camera]):
    n_cameras = len(cameras)
    camera_params = []
    model = cameras[0].model
    for cam in cameras:
        assert cam.model == model, "All cameras must have the same model"
        camera_params.extend(cam.get_camera_params())
    camera_params = np.array(camera_params)
    points_3d_df['point_index'] = points_3d_df.index
    points_df = points_2d_df.merge(points_3d_df, how='inner', on=['frame','label'], suffixes=('_cam',''))
    camera_indices = np.array(points_df['camera'], dtype=np.integer)
    point_indices = np.array(points_df['point_index'], dtype=np.integer)
    points_2d = np.array(points_df[['x_cam', 'y_cam']], dtype=np.float)
    points_3d = np.array(points_3d_df[['x', 'y', 'z']], dtype=np.float)
    n_points = len(points_3d)
    if model == 'standard':
        cost_func = standard_calib_cost_func
    elif model == 'fisheye':
        cost_func = fisheye_calib_cost_func
    else:
        raise InvalidCameraModelError(f"Camera model not supported: {model}")
    x0 = np.hstack((camera_params.ravel(), points_3d.ravel()))
    f0 = cost_func(x0, n_cameras, n_points, camera_indices, point_indices, points_2d)
    plt.plot(f0)
    A = create_calib_sba_sparsity_matrix(n_cameras, n_points, camera_indices, point_indices)
    t0 = time.time()
    res = least_squares(cost_func, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-6, method='trf',
                        args=(n_cameras, n_points, camera_indices, point_indices, points_2d), max_nfev=100)
    t1 = time.time()
    print("Optimization took {0:.0f} seconds".format(t1 - t0))
    plt.plot(f0)
    plt.plot(res.fun)
    plt.show()
    params = res.x
    camera_params = params[:n_cameras * 15].reshape((n_cameras, 15))
    for i, cam in enumerate(cameras):
        cam.set_from_camera_params(camera_params[i])
    new_points_3d = params[n_cameras * 15:].reshape((n_points, 3))
    new_points_3d_df = points_3d_df.copy()
    new_points_3d_df['x'] = new_points_3d[:, 0]
    new_points_3d_df['y'] = new_points_3d[:, 1]
    new_points_3d_df['z'] = new_points_3d[:, 2]
    return new_points_3d_df

def run_point_bundle_adjustment(points_3d_df, points_2d_df, cameras: List[Camera]):
    n_cameras = len(cameras)
    camera_params = []
    model = cameras[0].model
    for cam in cameras:
        assert cam.model == model, "All cameras must have the same model"
        camera_params.extend(cam.get_camera_params())
    camera_params = np.array(camera_params)
    points_3d_df['point_index'] = points_3d_df.index
    points_df = points_2d_df.merge(points_3d_df, how='inner', on=['frame','label'], suffixes=('_cam',''))
    camera_indices = np.array(points_df['camera'], dtype=np.integer)
    point_indices = np.array(points_df['point_index'], dtype=np.integer)
    points_2d = np.array(points_df[['x_cam', 'y_cam']], dtype=np.float)
    points_3d = np.array(points_3d_df[['x', 'y', 'z']], dtype=np.float)
    n_points = len(points_3d)
    if model == 'standard':
        cost_func = standard_point_cost_func
    elif model == 'fisheye':
        cost_func = fisheye_point_cost_func
    else:
        raise InvalidCameraModelError(f"Camera model not supported: {model}")
    
    x0 = points_3d.ravel()
    f0 = cost_func(x0, camera_params, n_cameras, n_points, camera_indices, point_indices, points_2d)
    plt.plot(f0)
    A = create_point_sba_sparsity_matrix(n_points, camera_indices, point_indices)
    t0 = time.time()
    res = least_squares(cost_func, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-6, method='trf',
                        args=(camera_params, n_cameras, n_points, camera_indices, point_indices, points_2d), max_nfev=100)
    t1 = time.time()
    print("Optimization took {0:.0f} seconds".format(t1 - t0))
    plt.plot(f0)
    plt.plot(res.fun)
    plt.show()
    params = res.x
    new_points_3d = params.reshape((n_points, 3))
    new_points_3d_df = points_3d_df.copy()
    new_points_3d_df['x'] = new_points_3d[:, 0]
    new_points_3d_df['y'] = new_points_3d[:, 1]
    new_points_3d_df['z'] = new_points_3d[:, 2]
    return new_points_3d_df
# ===================== END Sparse Bundle Adjustment Routines =======================



