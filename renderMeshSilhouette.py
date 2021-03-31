import pyrr
import glfw
import numpy as np
from math import sin, cos
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader

from lib.util.obj import ObjLoader
from lib.util.texture import load_texture


##############################################################################
# shaders
##############################################################################
vertex_src = """
# version 330

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec2 a_texture;
layout(location = 2) in vec3 a_normal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform mat4 normal_matrix;

out vec2 v_texture;
out vec3 v_normal;

void main()
{
    gl_Position = view * model * vec4(a_position, 1.0);
    v_texture = a_texture;
    v_normal = normalize(mat3(normal_matrix) * a_normal);       // using normal matrix
}
"""


fragment_src = """
# version 330

in vec2 v_texture;
in vec3 v_normal;

uniform vec3 camera_pos;
uniform vec3 camera_target;
uniform sampler2D s_texture;

out vec4 out_color;

void main()
{
    vec3 normal = normalize(v_normal);
    vec3 camera_dir = normalize(camera_pos - camera_target);
    float sil = dot(normal, camera_dir);

    if (sil < 0.2 && sil > -0.2)
        out_color = vec4(1.0, 1.0, 1.0, 1.0);
    else
        out_color = texture(s_texture, v_texture);
}
"""
##############################################################################
##############################################################################


# glfw callback functions
def window_resize(window, width, height):
    glViewport(0, 0, width, height)
    projection = pyrr.matrix44.create_perspective_projection_matrix(45, width / height, 0.1, 100)
    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)


##############################################################################
# glfw
##############################################################################
if not glfw.init():
    raise Exception("glfw can not be initialized!")
width, height = 1920, 1080
window = glfw.create_window(width, height, "Mesh Visualization", None, None)
if not window:
    glfw.terminate()
    raise Exception("glfw window can not be created!")
glfw.set_window_pos(window, 400, 200)
glfw.set_window_size_callback(window, window_resize)
glfw.make_context_current(window)
##############################################################################
##############################################################################


##############################################################################
# load model
##############################################################################
# mesh obj paths
face_obj_path = './assets/therock/Face.obj'
lefteye_obj_path = './assets/therock/LeftEye.obj'
righteye_obj_path = './assets/therock/RightEye.obj'

# mesh texture paths
face_tex_path = './assets/therock/textures/Texture_Face.jpg'
lefteye_tex_path = './assets/therock/textures/Texture_LeftEye.jpg'
righteye_tex_path = './assets/therock/textures/Texture_RightEye.jpg'

face_meta = ObjLoader.load_model(face_obj_path)
lefteye_meta = ObjLoader.load_model(lefteye_obj_path)
righteye_meta = ObjLoader.load_model(righteye_obj_path)

#================= FACE =================#
face_vertices = face_meta['v']
face_tex = face_meta['vt']
face_norms = face_meta['vn']

face_indices = face_meta['indices']
face_buffer = face_meta['buffer']

#================= EYES =================#
lefteye_indices = lefteye_meta['indices']
lefteye_buffer = lefteye_meta['buffer']

righteye_indices = righteye_meta['indices']
righteye_buffer = righteye_meta['buffer']

##############################################################################
##############################################################################


# compile the shader programs
shader = compileProgram(
    compileShader(vertex_src, GL_VERTEX_SHADER),
    compileShader(fragment_src, GL_FRAGMENT_SHADER)
)


##############################################################################
# VAO/VBO
##############################################################################
VAO = glGenVertexArrays(3)
VBO = glGenBuffers(3)

#================= FACE =================#
glBindVertexArray(VAO[0])

glBindBuffer(GL_ARRAY_BUFFER, VBO[0])
glBufferData(GL_ARRAY_BUFFER, face_buffer.nbytes, face_buffer, GL_STATIC_DRAW)

# face vertices (x, y, z)
glEnableVertexAttribArray(0)
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, face_buffer.itemsize * 8, ctypes.c_void_p(0))

# face textures (u, v)
glEnableVertexAttribArray(1)
glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, face_buffer.itemsize * 8, ctypes.c_void_p(12))

# face normals (x, y, z)
glEnableVertexAttribArray(2)
glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, face_buffer.itemsize * 8, ctypes.c_void_p(20))

glBindVertexArray(0)

#================= LEFT EYE =================#
glBindVertexArray(VAO[1])

glBindBuffer(GL_ARRAY_BUFFER, VBO[1])
glBufferData(GL_ARRAY_BUFFER, lefteye_buffer.nbytes, lefteye_buffer, GL_STATIC_DRAW)

# left eye vertices (x, y, z)
glEnableVertexAttribArray(0)
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, lefteye_buffer.itemsize * 8, ctypes.c_void_p(0))

# left eye textures (u, v)
glEnableVertexAttribArray(1)
glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, lefteye_buffer.itemsize * 8, ctypes.c_void_p(12))

# left eye normals (x, y, z)
glEnableVertexAttribArray(2)
glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, lefteye_buffer.itemsize * 8, ctypes.c_void_p(20))

glBindVertexArray(0)

#================= RIGHT EYE =================#
glBindVertexArray(VAO[2])

glBindBuffer(GL_ARRAY_BUFFER, VBO[2])
glBufferData(GL_ARRAY_BUFFER, righteye_buffer.nbytes, righteye_buffer, GL_STATIC_DRAW)

# left eye vertices (x, y, z)
glEnableVertexAttribArray(0)
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, righteye_buffer.itemsize * 8, ctypes.c_void_p(0))

# left eye textures (u, v)
glEnableVertexAttribArray(1)
glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, righteye_buffer.itemsize * 8, ctypes.c_void_p(12))

# left eye normals (x, y, z)
glEnableVertexAttribArray(2)
glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, righteye_buffer.itemsize * 8, ctypes.c_void_p(20))

glBindVertexArray(0)
##############################################################################
##############################################################################


##############################################################################
# textures
##############################################################################
textures = glGenTextures(3)
load_texture(face_tex_path, textures[0])
load_texture(lefteye_tex_path, textures[1])
load_texture(righteye_tex_path, textures[2])
##############################################################################
##############################################################################


##############################################################################
# setup/transformations
##############################################################################
glUseProgram(shader)

glClearColor(0, 0.1, 0.1, 1)
glEnable(GL_DEPTH_TEST)
glEnable(GL_BLEND)
glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

scale = pyrr.Matrix44.from_scale((0.005, 0.005, 0.005))
translation = pyrr.matrix44.create_from_translation(pyrr.Vector3([0.0, 0.0, 0.0]))
model = pyrr.matrix44.multiply(translation, scale)

projection = pyrr.matrix44.create_perspective_projection_matrix(
    fovy=45,
    aspect=width/height,
    near=0.1,
    far=1000
)

normal_matrix = np.linalg.inv(model).T

model_loc = glGetUniformLocation(shader, "model")
view_loc = glGetUniformLocation(shader, "view")
proj_loc = glGetUniformLocation(shader, "projection")
normal_matrix_loc = glGetUniformLocation(shader, "normal_matrix")

camera_pos_loc = glGetUniformLocation(shader, "camera_pos")
camera_target_loc = glGetUniformLocation(shader, "camera_target")

glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)
glUniformMatrix4fv(normal_matrix_loc, 1, GL_FALSE, normal_matrix)
##############################################################################
##############################################################################


##############################################################################
# main application loop
##############################################################################
while not glfw.window_should_close(window):
    # clear the buffers
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # move camera
    radius = 0.1
    camX = sin(0.5 * glfw.get_time()) * radius
    camZ = cos(0.5 * glfw.get_time()) * radius
    camera_position = pyrr.Vector3([camX, 0.0, camZ])
    camera_target = pyrr.Vector3([0.0, 0.0, 0.0])
    camera_up = pyrr.Vector3([0.0, 1.0, 0.0])
    view = pyrr.matrix44.create_look_at(
        eye=camera_position,
        target=camera_target,
        up=camera_up
    )

    #================= FACE =================#
    glBindVertexArray(VAO[0])
    glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)
    glUniform3fv(camera_pos_loc, 1, camera_position)
    glUniform3fv(camera_target_loc, 1, camera_target)
    glBindTexture(GL_TEXTURE_2D, textures[0])
    glDrawArrays(GL_TRIANGLES, 0, len(face_indices))
    glBindVertexArray(0)

    #================= EYES =================#
    glBindVertexArray(VAO[1])
    glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)
    glUniform3fv(camera_pos_loc, 1, camera_position)
    glUniform3fv(camera_target_loc, 1, camera_target)
    glBindTexture(GL_TEXTURE_2D, textures[1])
    glDrawArrays(GL_TRIANGLES, 0, len(lefteye_indices))
    glBindVertexArray(0)

    glBindVertexArray(VAO[2])
    glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)
    glUniform3fv(camera_pos_loc, 1, camera_position)
    glUniform3fv(camera_target_loc, 1, camera_target)
    glBindTexture(GL_TEXTURE_2D, textures[2])
    glDrawArrays(GL_TRIANGLES, 0, len(righteye_indices))
    glBindVertexArray(0)
   
    # swap front and back buffers | poll for and process events
    glfw.swap_buffers(window)
    glfw.poll_events()

glfw.terminate()
##############################################################################
##############################################################################
