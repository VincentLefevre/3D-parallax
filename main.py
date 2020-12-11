import numpy as np
import argparse
import glob
import os
from functools import partial
import vispy
import scipy.misc as misc
from tqdm import tqdm
import yaml
import time
import sys
from mesh import write_ply, read_ply, output_3d_photo
from utils import get_MiDaS_samples, read_MiDaS_depth
import torch
import cv2
from skimage.transform import resize
import imageio
import copy
from networks import Inpaint_Color_Net, Inpaint_Depth_Net, Inpaint_Edge_Net
from MiDaS.run import run_depth
from MiDaS.monodepth_net import MonoDepthNet
import MiDaS.MiDaS_utils as MiDaS_utils
from bilateral_filtering import sparse_bilateral_filtering
import dlib
from math import *
try:
    import cynetworkx as netx
except ImportError:
    import networkx as netx
from mesh import Canvas_view
from tkinter import *
from PIL import Image, ImageTk

def shape_to_np(shape, dtype="int"):
	# Initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)
	# Loop over the 68 facial landmarks and convert them to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	# Return the list of (x, y)-coordinates
	return coords

def init_profondeur():
    global detector, predictor,depth_ref, cap
    while (True):
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 1)
        if len(rects)>0 :
            break
    shape = predictor(gray, rects[0])
    shape = shape_to_np(shape)
    depth_ref = sqrt( (shape[42][0]-shape[39][0])**2 + (shape[42][1]-shape[39][1])**2 )

def affiche():
    global config, sample, detector, predictor, cap, num_col_webcam, num_row_webcam, verts, colors, faces, Height, Width, hFov, vFov, I
    global x_shift_range, y_shift_range, z_shift_range, border, depth, normal_canvas, all_canvas, mean_loc_depth, cam_mesh, image, depth_ref
    
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
    
    # Normalisation e[-1;-1] + Initialisation du repère caméra au centre (x :droite à gauche, y : bas en haut)
    x_norm = (2 * x_mean / num_col_webcam) - 1 
    y_norm = -(2 * y_mean / num_row_webcam) + 1
    if (depth_ref == 0) :
        z_norm = 0
    if (depth_ref != 0):
        z_norm = - sqrt( (shape[42][0]-shape[39][0])**2 + (shape[42][1]-shape[39][1])**2 ) / depth_ref + 1
        # Profondeur relative à l'initialisation de depth_ref
    
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

    return Img_3D

def next():
    global canvas
    img = cv2.imread('C:/Users/lucas/Documents/IOGS/3A/Projet_Parallax_3D/3d_photo_inpainting_master_mod/image/illan.jpg')
    img = cv2.resize(img,(400,400))
    b,g,r = cv2.split(img)
    img = cv2.merge((r,g,b))
    im = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=im)
    canvas.create_image(200,200,image=imgtk)
    canvas.image = imgtk
    print('NEXT')

def scanning():
    global canvas 
    if running == True :
        img = affiche()
        img = cv2.resize(img,(400,400))
        im = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=im)
        canvas.create_image(200,200,image=imgtk)
        canvas.image = imgtk
        app.update()
        scanning()

def Start() :
    global running
    running = True
    scanning()

def Stop():
    global running
    running = False

start_time = time.time()

## Config ##
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='argument.yml',help='Configure of post processing')
args = parser.parse_args()
config = yaml.load(open(args.config, 'r'))
if config['offscreen_rendering'] is True:
    vispy.use(app='egl')
os.makedirs(config['mesh_folder'], exist_ok=True)
os.makedirs(config['video_folder'], exist_ok=True)
os.makedirs(config['depth_folder'], exist_ok=True)
sample_list = get_MiDaS_samples(config['src_folder'], config['depth_folder'], config) #, config['specific'])
sample = sample_list[0]
depth = None
normal_canvas, all_canvas = None, None
if isinstance(config["gpu_ids"], int) and (config["gpu_ids"] >= 0):
    device = config["gpu_ids"]
else:
    device = "cpu"
mesh_fi = os.path.join(config['mesh_folder'], sample['src_pair_name'] +'.ply')
image = imageio.imread(sample['ref_img_fi'])

print(f"running on device {device}")
print("Image ==> ", sample['src_pair_name'])
print(f"Start")

if config['require_midas'] is True:
    run_depth([sample['ref_img_fi']], config['src_folder'], config['depth_folder'],
                config['MiDaS_model_ckpt'], MonoDepthNet, MiDaS_utils, target_w=640)
if 'npy' in config['depth_format']:
    config['output_h'], config['output_w'] = np.load(sample['depth_fi']).shape[:2]
else:
    config['output_h'], config['output_w'] = imageio.imread(sample['depth_fi']).shape[:2]
frac = config['longer_side_len'] / max(config['output_h'], config['output_w'])
config['output_h'], config['output_w'] = int(config['output_h'] * frac), int(config['output_w'] * frac)
config['original_h'], config['original_w'] = config['output_h'], config['output_w']
if image.ndim == 2:
    image = image[..., None].repeat(3, -1)
if np.sum(np.abs(image[..., 0] - image[..., 1])) == 0 and np.sum(np.abs(image[..., 1] - image[..., 2])) == 0:
    config['gray_image'] = True
else:
    config['gray_image'] = False
image = cv2.resize(image, (config['output_w'], config['output_h']), interpolation=cv2.INTER_AREA)
depth = read_MiDaS_depth(sample['depth_fi'], 3.0, config['output_h'], config['output_w'])
mean_loc_depth = depth[depth.shape[0]//2, depth.shape[1]//2]
if not(config['load_ply'] is True and os.path.exists(mesh_fi)):
    vis_photos, vis_depths = sparse_bilateral_filtering(depth.copy(), image.copy(), config, num_iter=config['sparse_iter'], spdb=False)
    depth = vis_depths[-1]
    model = None
    torch.cuda.empty_cache()
    print("Start Running 3D_Photo ...")
    print(f"Loading edge model at {time.time()-start_time}")
    depth_edge_model = Inpaint_Edge_Net(init_weights=True)
    depth_edge_weight = torch.load(config['depth_edge_model_ckpt'],
                                    map_location=torch.device(device))
    depth_edge_model.load_state_dict(depth_edge_weight)
    depth_edge_model = depth_edge_model.to(device)
    depth_edge_model.eval()

    print(f"Loading depth model at {time.time()-start_time}")
    depth_feat_model = Inpaint_Depth_Net()
    depth_feat_weight = torch.load(config['depth_feat_model_ckpt'],
                                    map_location=torch.device(device))
    depth_feat_model.load_state_dict(depth_feat_weight, strict=True)
    depth_feat_model = depth_feat_model.to(device)
    depth_feat_model.eval()
    depth_feat_model = depth_feat_model.to(device)
    print(f"Loading rgb model at {time.time()-start_time}")
    rgb_model = Inpaint_Color_Net()
    rgb_feat_weight = torch.load(config['rgb_feat_model_ckpt'],
                                    map_location=torch.device(device))
    rgb_model.load_state_dict(rgb_feat_weight)
    rgb_model.eval()
    rgb_model = rgb_model.to(device)
    graph = None

    print(f"Writing depth ply (and basically doing everything) at {time.time()-start_time}")
    rt_info = write_ply(image,
                            depth,
                            sample['int_mtx'],
                            mesh_fi,
                            config,
                            rgb_model,
                            depth_edge_model,
                            depth_edge_model,
                            depth_feat_model)
    rgb_model = None
    color_feat_model = None
    depth_edge_model = None
    depth_feat_model = None
    torch.cuda.empty_cache()
if config['save_ply'] is True or config['load_ply'] is True:
    print(f"Load ply at {time.time()-start_time}")
    verts, colors, faces, Height, Width, hFov, vFov = read_ply(mesh_fi)
    print(f"ply loaded at {time.time()-start_time}")
else:
    verts, colors, faces, Height, Width, hFov, vFov = rt_info

I = np.eye(4)
x_shift_range = config['x_shift_range'][0]
y_shift_range = config['y_shift_range'][0]
z_shift_range = config['z_shift_range'][0]
top = (config.get('original_h') // 2 - sample['int_mtx'][1, 2] * config['output_h'])
left = (config.get('original_w') // 2 - sample['int_mtx'][0, 2] * config['output_w'])
down, right = top + config['output_h'], left + config['output_w']
border = [int(xx) for xx in [top, down, left, right]]

## Dlib detection ## -- # Webcam : origine en haut à droite de notre pdv face à l'écran
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat') #entrainement
cap = cv2.VideoCapture(0)
num_col_webcam = cap.get(cv2.CAP_PROP_FRAME_WIDTH) 
num_row_webcam = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

## Render ## (Source - Outputphoto3d)
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
image = cv2.resize(image, (int(image.shape[1] / init_factor), int(image.shape[0] / init_factor)), interpolation=cv2.INTER_AREA)

### Interface / Sans Interface ###

if config['interface'] == True :
    
    running = True
    depth_ref = 0

    # Window
    global app
    app = Tk()
    app.title('Parallax 3D')
    app.geometry('800x700')
    app.minsize(200,400)
    app.iconbitmap("Logo.ico")
    app.config(background='#A9A9A9')

    # Frame
    frame = Frame(app,bg='#A9A9A9')

    # Right/Left Frame
    right_frame = Frame(frame,bg='#808080')
    right_frame.grid(row=0, column= 1,sticky=W,padx=10)
    left_frame = Frame(frame,bg='#808080')
    left_frame.grid(row=0, column= 0,sticky=W,padx=10)

    # Right_Frame 
    start_button = Button(right_frame, text='Start', font=('Arial',25), bg='#696969', fg='white', command = Start)
    stop_button = Button(right_frame, text='Stop', font=('Arial',25), bg='#696969', fg='white', command = Stop)
    profondeur_button = Button(right_frame, text='Set Depth', font=('Arial',25), bg='#696969', fg='white', command = init_profondeur)
    start_button.pack(pady=15, padx=10, fill=X)
    stop_button.pack(pady=15, padx=10, fill=X)
    profondeur_button.pack(pady=15, padx=10, fill=X)

    # Left_Frame
    img_name = [os.path.splitext(os.path.basename(xx))[0] for xx in glob.glob(os.path.join(config['src_folder'], '*' + config['img_format']))]
    img_path = os.path.join(config['src_folder'], img_name[0] +'.jpg')
    img = cv2.imread(img_path)
    img = cv2.resize(img,(400,400))
    b,g,r = cv2.split(img)
    img = cv2.merge((r,g,b))
    im = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=im)
    canvas = Canvas(left_frame,width=400,height=400,bg='#808080')
    canvas.create_image(200,200,image=imgtk)
    canvas.pack()

    #Afficher la frame
    frame.pack(expand=YES)

    # Afficher la fenetre
    app.mainloop()

else :
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
        
        P = []
        P.append(I * 1.)
        x_view, y_view, z_view = x_norm * x_shift_range, y_norm * y_shift_range, z_norm * z_shift_range
        P[-1][:3,-1] = np.array([x_view, y_view, z_view])

        ## Photo 3D ##
        Img_3D = output_3d_photo(verts.copy(), colors.copy(), faces.copy(), copy.deepcopy(Height), copy.deepcopy(Width), copy.deepcopy(hFov), copy.deepcopy(vFov), 
                            I, 
                            copy.deepcopy(sample['int_mtx']), 
                            config, 
                            P,
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
        cv2.imshow('Image',BGR_I)
        ##
        
        print('Step : ',time.time()-start_time)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
