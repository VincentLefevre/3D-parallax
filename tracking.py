import cv2
import numpy as np
import yaml
import argparse
from math import * 
import numpy as np
import dlib
import time
import os
from mesh import read_ply
import glob
import imageio
import copy
from mesh import output_3d_photo
from utils import read_MiDaS_depth
try:
    import cynetworkx as netx
except ImportError:
    import networkx as netx
from mesh import Canvas_view

def get_sample(image_folder, depth_folder, config):
    lines = [os.path.splitext(os.path.basename(xx))[0] for xx in glob.glob(os.path.join(image_folder, '*' + config['img_format']))]
    sample = []
    for seq_dir in lines:
        sample.append({})
        sdict = sample[-1]            
        sdict['depth_fi'] = os.path.join(depth_folder, seq_dir + config['depth_format'])
        sdict['ref_img_fi'] = os.path.join(image_folder, seq_dir + config['img_format'])
        H, W = imageio.imread(sdict['ref_img_fi']).shape[:2]
        sdict['int_mtx'] = np.array([[max(H, W), 0, W//2], [0, max(H, W), H//2], [0, 0, 1]]).astype(np.float32)
        if sdict['int_mtx'].max() > 1:
            sdict['int_mtx'][0, :] = sdict['int_mtx'][0, :] / float(W)
            sdict['int_mtx'][1, :] = sdict['int_mtx'][1, :] / float(H)
        #sdict['ref_pose'] = np.eye(4)
        sdict['tgt_name'] = [os.path.splitext(os.path.basename(sdict['depth_fi']))[0]]
        sdict['src_pair_name'] = sdict['tgt_name'][0]

    return sample


def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	# return the list of (x, y)-coordinates
	return coords


## Config ##
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='argument.yml',help='Configure of post processing')
args = parser.parse_args()
config = yaml.load(open(args.config, 'r'),yaml.Loader)

## Sample ##
sample_list = get_sample(config['src_folder'], config['depth_folder'], config)
sample = sample_list[0]

## Dlib detection ## -- # Webcam : origine en haut à droite de notre pdv face à l'écran
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('C:/Users/lucas/Documents/IOGS/3A/Projet_Parallax_3D/Test/shape_predictor_68_face_landmarks.dat') #entrainement
cap = cv2.VideoCapture(0)
num_col_webcam = cap.get(cv2.CAP_PROP_FRAME_WIDTH) 
num_row_webcam = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

## PLY
ply_file = "C:/Users/lucas/Documents/IOGS/3A/Projet_Parallax_3D/3d_photo_inpainting_master_mod2/mesh/moon.ply"
verts, colors, faces, Height, Width, hFov, vFov = read_ply(ply_file)

I = np.eye(4)
x_shift_range = config['x_shift_range'][0]
y_shift_range = config['y_shift_range'][0]
z_shift_range = config['z_shift_range'][0]
config['output_h'], config['output_w'] = imageio.imread(sample['depth_fi']).shape[:2]
frac = config['longer_side_len'] / max(config['output_h'], config['output_w'])
config['output_h'], config['output_w'] = int(config['output_h'] * frac), int(config['output_w'] * frac)
config['original_h'], config['original_w'] = config['output_h'], config['output_w']
top = (config.get('original_h') // 2 - sample['int_mtx'][1, 2] * config['output_h'])
left = (config.get('original_w') // 2 - sample['int_mtx'][0, 2] * config['output_w'])
down, right = top + config['output_h'], left + config['output_w']
border = [int(xx) for xx in [top, down, left, right]]
depth = None
normal_canvas, all_canvas = None, None
depth = read_MiDaS_depth(sample['depth_fi'], 3.0, config['output_h'], config['output_w'])
mean_loc_depth = depth[depth.shape[0]//2, depth.shape[1]//2]


## Render ##
cam_mesh = netx.Graph()
cam_mesh.graph['H'] = Height
cam_mesh.graph['W'] = Width
original_H = config.get('original_h'),
original_W = config.get('original_w'),
original_H = original_H[0]
original_W = original_W[0]
cam_mesh.graph['original_H'] = original_H
cam_mesh.graph['original_W'] = original_W
int_mtx = copy.deepcopy(sample['int_mtx'])
int_mtx_real_x = int_mtx[0] * Width
int_mtx_real_y = int_mtx[1] * Height
cam_mesh.graph['hFov'] = 2 * np.arctan((1. / 2.) * ((cam_mesh.graph['original_W']) / int_mtx_real_x[0]))
cam_mesh.graph['vFov'] = 2 * np.arctan((1. / 2.) * ((cam_mesh.graph['original_H']) / int_mtx_real_y[1]))
color = colors[..., :3]
fov_in_rad = max(cam_mesh.graph['vFov'], cam_mesh.graph['hFov'])
fov = (fov_in_rad * 180 / np.pi)
init_factor = 1
if config.get('anti_flickering') is True:
    init_factor = 3
if (cam_mesh.graph['original_H'] is not None) and (cam_mesh.graph['original_W'] is not None):
    canvas_w = cam_mesh.graph['original_W']
    canvas_h = cam_mesh.graph['original_H']
else:
    canvas_w = cam_mesh.graph['W']
    canvas_h = cam_mesh.graph['H']
canvas_size = max(canvas_h, canvas_w)
if normal_canvas is None:
    normal_canvas = Canvas_view(fov,
                                verts,
                                faces,
                                color,
                                canvas_size=canvas_size,
                                factor=init_factor,
                                bgcolor='gray',
                                proj='perspective')
else:
    normal_canvas.reinit_mesh(verts, faces, color)
    normal_canvas.reinit_camera(fov)

image = normal_canvas.render()
backup_img, backup_all_img, all_img_wo_bound = image.copy(), image.copy() * 0, image.copy() * 0
image = cv2.resize(image, (int(image.shape[1] / init_factor), int(image.shape[0] / init_factor)), interpolation=cv2.INTER_AREA)

n = 0

while(True):
    start_time = time.time()
    while (True):
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 1)
        if len(rects)>0 :
            break
    
    shape = predictor(gray, rects[0])
    shape = shape_to_np(shape)

    x_mean = floor((shape[39][0] + shape[42][0]) / 2)
    y_mean = floor((shape[39][1] + shape[42][1]) / 2)
    M = (x_mean,y_mean)

    if n == 0 :
        depth_ref = sqrt( (shape[42][0]-shape[39][0])**2 + (shape[42][1]-shape[39][1])**2 )
    n = 1

    # Normalisation e[-1;-1] + Initialisation du repère caméra au centre (x :droite à gauche, y : bas en haut)
    x_norm = (2 * x_mean / num_col_webcam) - 1 
    y_norm = -(2 * y_mean / num_row_webcam) + 1
    z_norm = - sqrt( (shape[42][0]-shape[39][0])**2 + (shape[42][1]-shape[39][1])**2 ) / depth_ref + 1
    # Profondeur relative à l'initialisation de depth_ref

    # -- Visualisation du cercle - pdv -- #
    cv2.circle(img, (x_mean, y_mean), 2, (0, 255, 0), -1)
    cv2.imshow('Placement des yeux', img)
    ## _______________________________ ##

    #A DETERMINER
    P = []
    P.append(I * 1.)
    x_view, y_view, z_view = x_norm * x_shift_range, y_norm * y_shift_range, z_norm * z_shift_range
    P[-1][:3,-1] = np.array([x_view, y_view, z_view])

    ## Photo 3D ##

    Img_3D = output_3d_photo(verts.copy(), colors.copy(), faces.copy(), copy.deepcopy(Height), copy.deepcopy(Width), copy.deepcopy(hFov), copy.deepcopy(vFov), 
                        I, 
                        copy.deepcopy(sample['int_mtx']), 
                        config, 
                        P, ##liste contenant les chemins pour chaque type de trajectoire
                        image,
                        cam_mesh,
                        config.get('original_h'), 
                        config.get('original_w'),
                        border=border,
                        normal_canvas=normal_canvas, 
                        all_canvas=all_canvas,
                        mean_loc_depth=mean_loc_depth)

    ## Affichage ##
    BGR_I = cv2.cvtColor(Img_3D, cv2.COLOR_RGB2BGR) # Open_CV -> Images BGR
    #cv2.circle(Img_3D,(y_mean*h,x_mean*l),2,(0,255,0),-1)
    cv2.imshow('Image',BGR_I)
    ##
    
    print('Step : ',time.time()-start_time)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
