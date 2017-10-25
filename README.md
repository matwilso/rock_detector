# rock_detector
[in progress]<br>
Domain randomization to train a convolutional nerual net (ResNet) rock detector for NASA Robotic Mining Competition

rock detector. exciting uh?


Check out this OpenAI blog post for an intro:
https://blog.openai.com/spam-detection-in-the-physical-world/


I followed the details in this paper:
https://arxiv.org/pdf/1703.06907.pdf


There was some nice stuff in OpenAI's [mujoco_py](https://github.com/openai/mujoco-py) repo to make this easier


# Mujoco Tips

- More documentation on lighting can be found here: http://www.glprogramming.com/red/chapter05.html#name10
- You need to call sim.step() to get the camera and light modders to update
- Read this: https://github.com/openai/mujoco-py/issues/148
- To make it so cameras don't look through walls, you need to add:

```
  <visual>
    <map znear=0.01 /> 
  </visual>
```


