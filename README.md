# rock_detector
[IN PROGRESS]<br>
Domain randomization to train a convolutional neural net (VGG16) rock 
detector using Tensorflow and Mujoco, for NASA Robotic Mining Competition

- [Intro](#intro)
- [Get Started + Installation](#install)
- [Mujoco Tips](#mujoco)


<a name="intro"></a>
# Introduction

Check out this OpenAI blog post for an intro to Domain Randomization:
https://blog.openai.com/spam-detection-in-the-physical-world/

I followed the details in this paper:
https://arxiv.org/pdf/1703.06907.pdf

There was some nice stuff in OpenAI's [mujoco_py](https://github.com/openai/mujoco-py) repo to make this easier


## Demo

Real robot view <br>
<img src="https://github.com/matwilso/rock_detector/blob/master/assets/practice.jpg?raw=true" alt="test" height="200">

YouTube video showing domain randomization <br>
[![](https://img.youtube.com/vi/KJBttWPo61E/0.jpg)](https://www.youtube.com/watch?v=KJBttWPo61E)

rock detector. exciting uh?


<a name="install"></a>
# Installation

## Install `mujoco_py`

READ [THIS](https://github.com/openai/mujoco-py/pull/145#issuecomment-356938564) if you are getting `ERROR: GLEW initialization error: Missing GL version`

~~This was a huge headache and I had to do some hacky stuff to get it to work on 
my Ubuntu desktop.  Mac was really easy to get it working and I would guess putting 
it in the cloud is not bad, because I'm pretty sure that's the way the folks at 
OpenAI are doing it.-~~

~~To get it to work, I cloned the repo, modified some source files, ran sudo python3 setup.py install in the folder and it somewhat worked.~~

```
git clone https://github.com/openai/mujoco-py.git
cd mujoco_py
pip3 install -r requirements.txt
sudo python3 setup.py install
```

~~Then, there will be many errors to fix, but most several are fixed by apt-get getting
or pip installing something.~~

~~After that, I was still getting some errors, like [this](https://github.com/openai/mujoco-py/issues/44).
I had to do some hacky stuff and I am not using GPU rendering, but it is running.
If you have any issues, feel free to reach out to me.~~



## Install `blender`

```
sudo apt-get install blender
```

Test it with 
```
blender
```

### If you are running on your local desktop (w/ monitor):

Use the file in `assets/add_mesh_rock.zip` and follow:
https://blender.stackexchange.com/questions/8746/how-can-i-make-unique-rocks-in-blender-without-having-to-model-them-by-hand

Then activate it by clicking the check mark

### [untested] Alternatively, if you are running headless (in cloud):
```
cp /path/to/rock_detector/assets/add_mesh_rocks.zip ~/.config/blender/<VERSION>/scripts/addons/

cd ~/.config/blender/<VERSION>/scripts/addons
unzip add_mesh_rocks.zip
```


<a name="mujoco"></a>
# Mujoco Tips

- You need to call sim.forward() or sim.step() to get the camera and light modders to update
- You can't scale a mesh after it has been loaded (http://www.mujoco.org/forum/index.php?threads/how-to-scale-already-exist-model.3483/)
- Read this: https://github.com/openai/mujoco-py/issues/148 and this: https://github.com/openai/gym/issues/234
- The maximum number of lights that can be active simultaneously is 8, counting the headlight
- More documentation on lighting can be found here: http://www.glprogramming.com/red/chapter05.html#name10
- To make it so cameras don't look through walls, you need to add:

```
  <visual>
    <map znear=0.01 /> 
  </visual>
```


