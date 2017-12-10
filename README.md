# rock_detector
[in progress]<br>
Domain randomization to train a convolutional nerual net (ResNet) rock detector for NASA Robotic Mining Competition





Initial demo randomization (still need to get the rocks better)

![alt text](/assets/demo.png)


rock detector. exciting uh?


Check out this OpenAI blog post for an intro:
https://blog.openai.com/spam-detection-in-the-physical-world/


I followed the details in this paper:
https://arxiv.org/pdf/1703.06907.pdf


There was some nice stuff in OpenAI's [mujoco_py](https://github.com/openai/mujoco-py) repo to make this easier


# Installation

## Install `mujoco_py`

This was a huge headache and I had to do some hacky stuff to get it to work
on my Ubuntu desktop.  Mac was really easy to get it working and I think
putting it in the cloud is not bad because they are headless.  But
Ubuntu desktop.

I cloned the repo, modified some source files, ran sudo python3 setup.py install in the folder and it somewhat worked.
```
git clone https://github.com/openai/mujoco-py.git

cd mujoco_py

pip3 install -r requirements.txt

sudo python3 setup.py install 
```

There will be many errors to fix, but most are installed by apt-get getting
or pip installing something.


## Install `blender`

```
sudo apt-get install blender
```

Test it with 
```
blender
```

Open blender, run 








# Mujoco Tips

- More documentation on lighting can be found here: http://www.glprogramming.com/red/chapter05.html#name10
- You need to call sim.step() to get the camera and light modders to update
- You can't scale a mesh after it has been loaded (http://www.mujoco.org/forum/index.php?threads/how-to-scale-already-exist-model.3483/)
- Read this: https://github.com/openai/mujoco-py/issues/148 and this: https://github.com/openai/gym/issues/234
- To make it so cameras don't look through walls, you need to add:

```
  <visual>
    <map znear=0.01 /> 
  </visual>
```


