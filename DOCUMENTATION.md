# Documentation

## Python scripts

These files are for our monocular 3D Tracking pipeline:

`main.py` Reproduces parallax/depth effect by tracking user in real-time via his webcam using a single source image

`mesh.py` Functions about context-aware depth inpainting

`mesh_tools.py` Some common functions used in `mesh.py`

`utils.py` Some common functions used in image preprocessing, data loading

`networks.py` Network architectures of inpainting model


MiDaS/

`run.py` Execute depth estimation

`monodepth_net.py` Network architecture of depth estimation model

`MiDaS_utils.py` Some common functions in depth estimation


## Configuration

```bash
argument.yml
```
- `require_midas: True`
    - Set it to `True` if the user wants to use depth map estimated by `MiDaS`.
    - Set it to `False` if the user wants to use manually edited depth map.
    - If the user wants to edit the depth (disparity) map manually, we provide `.png` format depth (disparity) map.
    - Remember to switch this parameter from `True` to `False` when using manually edited depth map.
- `load_ply: False`
    - Action to load existed mesh (.ply) file
- `save_ply: True`
    - Action to store the output mesh (.ply) file
    - Disable this option `save_ply: False` to reduce the computational time.
- `Interface: True`
    - Set it to `True` if you want an interface to manualy activate/set the depth origin and the start and stop buttons 
    - Set it to `False` if you only want the window of the tracked face and the output window 
- `depth_edge_model_ckpt: checkpoints/EdgeModel.pth`
    - Pretrained model of depth-edge inpainting
- `depth_feat_model_ckpt: checkpoints/DepthModel.pth`
    - Pretrained model of depth inpainting
- `rgb_feat_model_ckpt: checkpoints/ColorModel.pth`
    - Pretrained model of color inpainting
- `MiDaS_model_ckpt: MiDaS/model.pt`
    - Pretrained model of depth estimation
- `x_shift_range: [0.04]`
    - Set the amplitude of the translation on x-axis.
    - Must be positive, typical range in order not to have unsatisfactory inpainting is around 0.05 depending on the given picture
    - This parameter is stored in a list.
- `y_shift_range: [0.02]`
    - Set the amplitude of the translation on y-axis.
    - Must be positive, typical range in order not to have unsatisfactory inpainting is around 0.05 depending on the given picture
    - This parameter is stored in a list.
- `z_shift_range: [0.05]`
    - Set the amplitude of the translation on z-axis.
    - Must be positive, typical range in order not to have unsatisfactory inpainting is around 0.1 depending on the given picture
    - This parameter is stored in a list.
- `longer_side_len: 960`
    - The length of larger dimension in output resolution.
- `src_folder: image`
    - Input image directory. 
- `depth_folder: depth`
    - Estimated depth directory.
- `mesh_folder: mesh`
    - Output 3-D mesh directory.
- `video_folder: video`
    - Output rendered video directory
- `inference_video: True`
    - Action to rendered the output video
- `gpu_ids: 0`
    - The ID of working GPU. Leave it blank or negative to use CPU. Specifically if your GPU is not compatible with CUDA (check here: https://developer.nvidia.com/cuda-gpus)
- `offscreen_rendering: True`
    - If you're executing the process in a remote server (via ssh), please switch on this flag. 
    - Sometimes, using off-screen rendering result in longer execution time.
- `img_format: '.jpg'`
    - Input image format.
- `depth_format: '.npy'`
    - Input depth (disparity) format. Use NumPy array file as default.
    - If the user wants to edit the depth (disparity) map manually, we provide `.png` format depth (disparity) map.
        - Remember to switch this parameter from `.npy` to `.png` when using depth (disparity) map with `.png` format.
- `depth_threshold: 0.04`
    - A threshold in disparity, adjacent two pixels are discontinuity pixels 
      if the difference between them excceed this number.
- `ext_edge_threshold: 0.002`
    - The threshold to define inpainted depth edge. A pixel in inpainted edge 
      map belongs to extended depth edge if the value of that pixel exceeds this number,
- `sparse_iter: 5`
    - Total iteration numbers of bilateral median filter
- `filter_size: [7, 7, 5, 5, 5]`
    - Window size of bilateral median filter in each iteration.
- `sigma_s: 4.0`
    - Intensity term of bilateral median filter
- `sigma_r: 0.5`
    - Spatial term of bilateral median filter
- `redundant_number: 12`
    - The number defines short segments. If a depth edge is shorter than this number, 
      it is a short segment and removed.
- `background_thickness: 70`
    - The thickness of synthesis area.
- `context_thickness: 140`
    - The thickness of context area.
- `background_thickness_2: 70`
    - The thickness of synthesis area when inpaint second time.
- `context_thickness_2: 70`
    - The thickness of context area when inpaint second time.
- `discount_factor: 1.00`
- `log_depth: True`
    - The scale of depth inpainting. If true, performing inpainting in log scale. 
      Otherwise, performing in linear scale.
- `largest_size: 512`
    - The largest size of inpainted image patch.
- `depth_edge_dilate: 10`
    - The thickness of dilated synthesis area.
- `depth_edge_dilate_2: 5`
    - The thickness of dilated synthesis area when inpaint second time.
- `extrapolate_border: True`
    - Action to extrapolate out-side the border.
- `extrapolation_thickness: 60`
    - The thickness of extrapolated area.
- `repeat_inpaint_edge: True`
    - Action to apply depth edge inpainting model repeatedly. Sometimes inpainting depth 
      edge once results in short inpinated edge, apply depth edge inpainting repeatedly 
      could help you prolong the inpainted depth edge. 
- `crop_border: [0.03, 0.03, 0.05, 0.03]`
    - The fraction of pixels to crop out around the borders `[top, left, bottom, right]`.
- `anti_flickering: True`
    - Action to avoid flickering effect in the output video. 
    - This may result in longer computational time in rendering phase.
