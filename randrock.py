from add_mesh_rocks import rockgen
import bpy

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()
bpy.ops.mesh.rocks()
bpy.data.objects['rock'].location.x = 0
bpy.data.objects['rock'].location.y = 0
bpy.data.objects['rock'].location.z = 0
bpy.ops.export_mesh.stl(filepath="ocean_floor.stl", check_existing=False)


