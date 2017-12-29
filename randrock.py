from add_mesh_rocks import rockgen
import bpy
#import xml.etree.ElementTree as ET
import json
import ast
from random import uniform


def make_settings():
    """Parse the converted json file for settings"""
    settings = {}
    num_of_rocks = 1

    obj = json.load(open('assets/add_mesh_rocks.json'))
    presets = [obj["settings"]["default"]] + obj["settings"]["preset"]

    for preset in presets:
        title = preset["title"]
        # SIZE
        size = preset["size"]

        x, y, z = size["scale"]
        if title == "Default":
            scale = uniform(float(x["lower"]), float(x["upper"]))
            scale_X = [scale, scale]
            scale_Y = [scale, scale]
            scale_Z = [scale, scale]
        else:
            scale_X = [float(x["lower"]), float(x["upper"])]
            scale_Y = [float(y["lower"]), float(y["upper"])]
            scale_Z = [float(z["lower"]), float(z["upper"])]

        x, y, z = size["skew"]
        skew_X = float(x["value"])
        skew_Y = float(y["value"])
        skew_Z = float(z["value"])

        scale_fac = ast.literal_eval(size["scale_fac"])
        use_scale_dis = bool(size["use_scale_dis"])

        # SHAPE
        shape = preset["shape"]

        deform = float(shape["deform"])
        rough = float(shape["rough"])
        detail = float(shape["detail"])
        display_detail = float(shape["display_detail"])
        smooth_fac = float(shape["smooth_fac"])
        smooth_it = float(shape["smooth_it"])


        # MATERIAL
        material = preset["material"]
        
        mat_enable = bool(material["mat_enable"])
        mat_color = ast.literal_eval(material["mat_color"])
        mat_bright = float(material["mat_bright"])
        mat_rough = float(material["mat_rough"])
        mat_spec = float(material["mat_spec"])
        mat_hard = float(material["mat_hard"])
        mat_use_trans = bool(material["mat_use_trans"])
        mat_alpha = float(material["mat_alpha"])
        mat_cloudy = float(material["mat_cloudy"])
        mat_IOR = float(material["mat_IOR"])
        mat_mossy = float(material["mat_mossy"])

        # RANDOM
        random = preset["random"]

        use_generate = bool(random["use_generate"])
        use_random_seed = bool(random["use_random_seed"])
        user_seed = float(random["user_seed"])


        settings[title] = [
                          context,
                          scale_X,
                          skew_X,
                          scale_Y,
                          skew_Y,
                          scale_Z,
                          skew_Z,
                          scale_fac,
                          detail,
                          display_detail,
                          deform,
                          rough,
                          smooth_fac,
                          smooth_it,
                          mat_enable,
                          mat_color,
                          mat_bright,
                          mat_rough,
                          mat_spec,
                          mat_hard,
                          mat_use_trans,
                          mat_alpha,
                          mat_cloudy,
                          mat_IOR,
                          mat_mossy,
                          num_of_rocks,
                          user_seed,
                          False,
                          use_random_seed
        ]

    return settings

context = bpy.context
settings = make_settings()

for i in range(3):
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    rockgen.generateRocks(*settings["Default"])
    
    #import pdb; pdb.set_trace()
    #print(bpy.data.objects)

    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY")
    rock_name = bpy.data.objects.keys()[0]
    bpy.data.objects[rock_name].location.x = 0
    bpy.data.objects[rock_name].location.y = 0
    bpy.data.objects[rock_name].location.z = 0
    
    #bpy.ops.object.modifier_add(type='DECIMATE')
    #bpy.data.objects["rock"].modifiers["Decimate"].ratio = 0.1
    #bpy.ops.object.modifier_apply(apply_as="DATA")
    
    bpy.ops.export_mesh.stl(filepath="xmls/nasa/meshes/rock{}.stl".format(i), check_existing=False)


bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

rockgen.generateRocks(*settings["Fake Ocean"])
bpy.ops.export_mesh.stl(filepath="xmls/nasa/meshes/dirt.stl", check_existing=False)


