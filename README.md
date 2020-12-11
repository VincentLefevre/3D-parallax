
We offer an experience of 3D on a single 2D image using the parallax effect, i.e, the user is able move his real-time tracked face to visualize the depth effect. By using an inpainting method, the program is able to paint the missing background which is now accessible by the user's movements. 

The inpainting method and a great part of the code have been taken from: 
\begin{filecontents}{References.bib}
@inproceedings{Shih3DP20,
  author = {Shih, Meng-Li and Su, Shih-Yang and Kopf, Johannes and Huang, Jia-Bin},
  title = {3D Photography using Context-aware Layered Depth Inpainting},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2020}
}
\end{filecontents}

@inproceedings{Shih3DP20,
  author = {Shih, Meng-Li and Su, Shih-Yang and Kopf, Johannes and Huang, Jia-Bin},
  title = {3D Photography using Context-aware Layered Depth Inpainting},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2020}
}

The eye detection is based on the shape predictor of 68 face landmarks from dlib. At first, we followed precisely the user's pupils. Then, given the precision needed and thanks to the dectector, we decided to track the point in the middle between the two eyes.

<br/>

## Prerequisites

- Anaconda
- Python 3.7 (tested on 3.7.4)
- PyTorch 1.4.0 (tested on 1.4.0 for execution)

and the Python dependencies listed in [requirements.txt](requirements.txt)
- To get started, please run the following commands:
    ```bash
    conda create -n 3DP python=3.7 anaconda
    conda activate 3DP
    ```
    Add Path environment variable to the environement 3DP

    ```bahs
    conda install git
    pip install Cython==3.0a5
    pip install git+https://github.com/pattern-inc/cynetworkx.git
    pip install -r requirements.txt
    conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit==10.1.243 -c pytorch
    ```
- Next, please download the model weight using the following command:
    Install Cygwin with package wget (on Windows) and in the Cygwin terminal run the following commands :
    ```bash
    chmod +x download.sh
    ./download.sh
    ```    

## Quick start

Please follow the instructions in this section. 
This should allow to execute our results.
For more detailed instructions, please refer to [`DOCUMENTATION.md`](DOCUMENTATION.md).

## Execute

1. Put one ```.jpg``` file (e.g., test.jpg) into the ```image``` folder. 
    - E.g., `image/moon.jpg`
2. Run the following command in prompt or configure your favorite IDE 
    ```bash
    python main.py --config argument.yml
    ```
    - Note: The 3D depth file (.ply) process usually takes about 2-3 minutes depending on the available computing resources.
    - Note: Do verify that your IDE has webcam access in your parameters. 
3. The results are stored in the following directories:
    - Corresponding depth map estimated by [MiDaS](https://github.com/intel-isl/MiDaS.git) 
        - E.g. ```depth/moon.npy```, ```depth/moon.png```
        - User could edit ```depth/moon.png``` manually. 
            - Remember to set the following two flags as listed below if user wants to use manually edited ```depth/moon.png``` as input for 3D Photo.
                - `depth_format: '.png'`
                - `require_midas: False`
    - Inpainted 3D mesh (Optional: User need to switch on the flag `save_ply`)
        - E.g. ```mesh/moon.ply```
    - Note: We assume that the object of focus is located at the center of the image.
4. Depending on the flag `interface`, an interface is open with the display and buttons to start/stop the display and start/set depth feeling.
5. (Optional) If you want to change the default configuration. Please read [`DOCUMENTATION.md`](DOCUMENTATION.md) and modified ```argument.yml```.

## Food for thought

- At first, we implemented a method that pre-stored all the frames displayed for each user's point of view. It worked finely but required too much memory. 

- In this project to compute the depth we normalized the distance between the two eyes of the user by a given distance (either the distance on the first frame for the case `interface : False` or the distance when the `set depth` button is pressed for the case `interface : True`). However, another method would be to compute the real distance of the user to his screen given the webcam intrinsic parameters (accessible by classical camera calibration). 

## License
This work is licensed under MIT License. See [LICENSE](LICENSE) for details. 

If you find our code/models useful, please consider citing us:
```
@inproceedings{
  author = {Vincent Lefevre, Lucas Pometti, Alexandre Labattut},
  title = {Parallax 3D},
  booktitle = {Project supervised by Bertrand Simon at the Institut Optique Graduate School},
  year = {2020}
}
```
