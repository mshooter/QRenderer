from __future__ import annotations

import numpy as np
import sys 
from textwrap import dedent
import trimesh
from PIL import Image

from PySide6.QtCore import (QSize, Qt, QTimer)
from PySide6.QtGui import (QMatrix4x4, QVector3D, QOpenGLContext, QSurfaceFormat, QWindow, QDoubleValidator)
from PySide6.QtOpenGL import (QOpenGLBuffer, QOpenGLShader,
                              QOpenGLShaderProgram, QOpenGLTexture)
from PySide6.QtWidgets import (QApplication, QHBoxLayout, QVBoxLayout, QMessageBox,
                               QWidget, QSlider, QLabel, QGroupBox,  QFormLayout, QPushButton, QLineEdit )
from PySide6.support import VoidPtr
try:
    from OpenGL import GL
except ImportError:
    app = QApplication(sys.argv)
    message_box = QMessageBox(QMessageBox.Critical, "ContextInfo",
                              "PyOpenGL must be installed to run this example.", QMessageBox.Close)
    message_box.setDetailedText("Run:\npip install PyOpenGL PyOpenGL_accelerate")
    message_box.exec()
    sys.exit(1)

vertex_shader_source = dedent("""
#version 110
attribute vec3 posAttr;
attribute vec3 normalAttr;
attribute vec2 texCoordAttr;
varying vec2 texCoord;
varying vec3 fragNormal;
varying vec3 fragPos;
uniform mat4 matrix;
uniform mat3 normalMatrix;
void main() {
    gl_Position = matrix * vec4(posAttr, 1.0);
    texCoord = texCoordAttr;
    fragNormal = normalize(normalMatrix * normalAttr);  // Transform normal to view space
    fragPos = vec3(matrix * vec4(posAttr, 1.0));  // Transform vertex position to view space
}
    """)

fragment_shader_source = dedent("""
#version 110
varying vec2 texCoord;
varying vec3 fragNormal;
varying vec3 fragPos;
uniform sampler2D textureSampler;
uniform vec3 lightPos;
uniform vec3 viewPos;
uniform vec3 lightColor;
uniform vec3 objectColor;

void main() {
    vec4 texColor = texture2D(textureSampler, texCoord);

    // Ambient
    float ambientStrength = 0.1;
    vec3 ambient = ambientStrength * lightColor;

    // Diffuse
    vec3 norm = normalize(fragNormal);
    vec3 lightDir = normalize(lightPos - fragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;

    // Specular
    float specularStrength = 0.5;
    vec3 viewDir = normalize(viewPos - fragPos);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0); // Update to 32.0
    vec3 specular = specularStrength * spec * lightColor;

    // Combine results
    vec3 result = (ambient + diffuse + specular) * texColor.rgb * objectColor;
    gl_FragColor = vec4(result, texColor.a);
}
    """)

# Load the mesh and extract vertices, faces, and texture coordinates
#mesh = trimesh.load_mesh("smal/my_smpl_39dogsnorm_Jr_4_dog_remesh4000.obj")
mesh = trimesh.load_mesh("cube.obj")
vertices = mesh.vertices[mesh.faces].reshape(-1, 3).astype(np.float32)
faces = mesh.faces.astype(np.uint32)
# Access vertex normals directly
# if mesh.vertex_normals is not None:
#     normals = mesh.vertex_normals.astype(np.float32)  # Access vertex normals
# else:
    # print("No vertex normals found in the mesh, so we are calculating it.")
# Calculate vertex normals
mesh.vertex_normals = trimesh.geometry.mean_vertex_normals(len(mesh.vertices), mesh.faces, mesh.face_normals)
normals = mesh.vertex_normals[mesh.faces].reshape(-1, 3).astype(np.float32)

tex_coords = mesh.visual.uv[mesh.faces].reshape(-1, 2).astype(np.float32)

class RenderWindow(QWindow):
    def __init__(self, fmt):
        super().__init__()
        self.setSurfaceType(QWindow.OpenGLSurface)
        self.setFormat(fmt)
        self.context = QOpenGLContext(self)
        self.context.setFormat(self.requestedFormat())
        if not self.context.create():
            raise Exception("Unable to create GL context")
        self.program = None
        self.angle = 0
        self.lightPosition = QVector3D(0,0,0)
        self.objectColor = QVector3D(0.5, 0.5, 0.5)
        self.texture = None
        self.timer = None
        self.rotation = QVector3D() 
        self.aspect_ratio = 4/3
        self.zoom = -5.0  # Initial zoom level
        
    def setRot(self, val, is_x, is_y, is_z):
        if is_x: 
            self.rotation.setX(np.pi * val)
        elif is_y:
            self.rotation.setY(np.pi * val)
        elif is_z:
            self.rotation.setZ(np.pi * val)
        self.render()

    def setObjectColor_r(self, r):
        r = float(r)
        self.objectColor.setX(r)
        self.render()

    def setObjectColor_g(self, g):
        g = float(g)
        self.objectColor.setY(g)
        self.render()

    def setObjectColor_b(self, b):
        b = float(b)
        self.objectColor.setZ(b)
        self.render()

    def setZoom(self, val):
        self.zoom = -val / 10.0  # Convert slider value to a suitable zoom range
        self.render()

    def init_gl(self):
        self.program = QOpenGLShaderProgram(self)
        self.vbo = QOpenGLBuffer()

        if not self.program.addShaderFromSourceCode(QOpenGLShader.Vertex, vertex_shader_source):
            raise Exception(f"Vertex shader could not be added: {self.program.log()}")
        if not self.program.addShaderFromSourceCode(QOpenGLShader.Fragment, fragment_shader_source):
            raise Exception(f"Fragment shader could not be added: {self.program.log()}")
        if not self.program.link():
            raise Exception(f"Could not link shaders: {self.program.log()}")

        self._pos_attr = self.program.attributeLocation("posAttr")
        self._normal_attr = self.program.attributeLocation("normalAttr")
        self._tex_coord_attr = self.program.attributeLocation("texCoordAttr")
        self._matrix_uniform = self.program.uniformLocation("matrix")
        self._texture_uniform = self.program.uniformLocation("textureSampler")
        self._light_pos_uniform = self.program.uniformLocation("lightPos")
        self._view_pos_uniform = self.program.uniformLocation("viewPos")
        self._light_color_uniform = self.program.uniformLocation("lightColor")
        self._object_color_uniform = self.program.uniformLocation("objectColor")
        self._normal_matrix_uniform = self.program.uniformLocation("normalMatrix")

        self.vbo.create()
        self.vbo.bind()
        vertices_data = vertices.tobytes()
        tex_coords_data = tex_coords.tobytes()
        normals_data = normals.tobytes()
        total_bytes = len(vertices_data) + len(normals_data) + len(tex_coords_data)

        self.vbo.allocate(total_bytes)
        self.vbo.write(0, VoidPtr(vertices_data), len(vertices_data))
        self.vbo.write(len(vertices_data), VoidPtr(normals_data), len(normals_data))
        self.vbo.write(len(vertices_data) + len(normals_data), VoidPtr(tex_coords_data), len(tex_coords_data))
        # --- 
        #self.vbo.allocate(VoidPtr(vertices_data), len(vertices_data) + len(tex_coords_data))
        #self.vbo.write(len(vertices_data), VoidPtr(tex_coords_data), len(tex_coords_data))
        #self.vbo.allocate(VoidPtr(vertices_data), len(vertices_data) )
        #self.vbo.write(len(vertices_data), VoidPtr(vertices_data), len(vertices_data))
        self.vbo.release()

        # Load texture
        self.texture = QOpenGLTexture(QOpenGLTexture.Target2D)
        #image = Image.open("dog_B_posed_0003.png")
        image = Image.open("cube.jpeg")
        image = image.convert("RGBA")
        img_data = image.tobytes("raw", "RGBA", 0, -1)
        self.texture.setSize(image.width, image.height)
        self.texture.setFormat(QOpenGLTexture.RGBA8_UNorm)
        self.texture.allocateStorage()
        self.texture.setData(QOpenGLTexture.RGBA, QOpenGLTexture.UInt8, img_data)
        self.texture.setMinificationFilter(QOpenGLTexture.Linear)
        self.texture.setMagnificationFilter(QOpenGLTexture.Linear)

    def setup_vertex_attribs(self):
        self.vbo.bind()
        self.program.setAttributeBuffer(self._pos_attr, GL.GL_FLOAT, 0, 3, 0)
        self.program.setAttributeBuffer(self._normal_attr, GL.GL_FLOAT, vertices.nbytes, 3, 0)
        self.program.setAttributeBuffer(self._tex_coord_attr, GL.GL_FLOAT, vertices.nbytes + normals.nbytes, 2, 0)
        self.program.enableAttributeArray(self._pos_attr)
        self.program.enableAttributeArray(self._normal_attr)
        self.program.enableAttributeArray(self._tex_coord_attr)
        self.vbo.release()

    def exposeEvent(self, event):
        if self.isExposed():
            self.render()
            # if self.timer is None:
            #     self.timer = QTimer(self)
            #     self.timer.timeout.connect(self.slot_timer)
            # if not self.timer.isActive():
            #     self.timer.start(10)
        else: 
            if self.timer and self.timer.isActive():
                self.timer.stop()

    def slot_timer(self):
        radius = 3.0
        self.angle += np.radians(1); 
        self.lightPosition.setX(radius * np.cos(self.angle)) 
        self.lightPosition.setZ(radius * np.sin(self.angle)) 
        self.lightPosition.setY(2.0) 
        self.render()

    def resizeEvent(self, event):
        width = event.size().width() 
        height = event.size().height() 
        self.aspect_ratio = width / height if height != 0 else 1
        self.render()

    def render(self):
        if not self.context.makeCurrent(self):
            raise Exception("makeCurrent() failed")
        functions = self.context.functions()
        functions.glEnable(GL.GL_DEPTH_TEST)
        functions.glEnable(GL.GL_CULL_FACE)  # Enable face culling
        functions.glCullFace(GL.GL_BACK)  # Cull back faces
        functions.glClearColor(0.5, 0.5, 0.5, 1)  # Light gray background
        functions.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        if self.program is None:
            self.init_gl()

        retina_scale = self.devicePixelRatio()
        functions.glViewport(0, 0, self.width() * retina_scale, self.height() * retina_scale)

        self.program.bind()
        matrix = QMatrix4x4()
        matrix.perspective(45, self.aspect_ratio, 0.1, 100)
        matrix.translate(0, 0, self.zoom)
        matrix.rotate(self.rotation.x(), 1, 0, 0)
        matrix.rotate(self.rotation.y(), 0, 1, 0)
        matrix.rotate(self.rotation.z(), 0, 0, 1)
        normal_matrix = matrix.normalMatrix()  # Extract the normal matrix
    
        self.program.setUniformValue(self._matrix_uniform, matrix)
        self.program.setUniformValue(self._normal_matrix_uniform, normal_matrix)
        self.program.setUniformValue(self._light_pos_uniform, self.lightPosition)
        self.program.setUniformValue(self._view_pos_uniform, QVector3D(0.0, 0.0, self.zoom))
        self.program.setUniformValue(self._light_color_uniform, QVector3D(1.0, 1.0, 1.0))
        self.program.setUniformValue(self._object_color_uniform, self.objectColor)  # Example object color

        self.texture.bind()
        self.program.setUniformValue(self._texture_uniform, 0)

        self.setup_vertex_attribs()
        functions.glDrawArrays(GL.GL_TRIANGLES, 0, len(vertices))

        self.program.release()
        self.context.swapBuffers(self)
        self.context.doneCurrent()

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("QRenderer")
        layout = QVBoxLayout(self)
        self._render_window = RenderWindow(QSurfaceFormat())
        container = QWidget.createWindowContainer(self._render_window)
        container.setMinimumSize(QSize(400, 400))

        controls_layout = QHBoxLayout()
        slider_group = QGroupBox("Moving Controls")
        slider_layout = QVBoxLayout()

        # Create Object Colour Group Box
        translation_group = QGroupBox("Base Colour")
        translation_layout = QHBoxLayout()
        translation_x = QLineEdit()
        translation_x.setAlignment(Qt.AlignCenter)
        translation_x.setValidator(QDoubleValidator(
                0.0, # bottom
                100.0, # top
                6, # decimals 
                notation=QDoubleValidator.StandardNotation
            ))
        translation_x.setText("0.5")
        translation_y = QLineEdit()
        translation_y.setAlignment(Qt.AlignCenter)
        translation_y.setValidator(QDoubleValidator(
                0.0, # bottom
                100.0, # top
                6, # decimals 
                notation=QDoubleValidator.StandardNotation
            ))
        translation_y.setText("0.5")
        translation_z = QLineEdit()
        translation_z.setAlignment(Qt.AlignCenter)
        translation_z.setValidator(QDoubleValidator(
                0.0, # bottom
                100.0, # top
                6, # decimals 
                notation=QDoubleValidator.StandardNotation
            ))
        translation_z.setText("0.5")
        
        translation_x.textChanged.connect(self._render_window.setObjectColor_r)
        translation_y.textChanged.connect(self._render_window.setObjectColor_g)
        translation_z.textChanged.connect(self._render_window.setObjectColor_b)
        translation_layout.addWidget(QLabel("X:"))
        translation_layout.addWidget(translation_x)
        translation_layout.addWidget(QLabel("Y:"))
        translation_layout.addWidget(translation_y)
        translation_layout.addWidget(QLabel("Z:"))
        translation_layout.addWidget(translation_z)
        translation_group.setLayout(translation_layout)
        # self.apply_button = QPushButton("Apply Transformations")
        # self.apply_button.clicked.connect(self.apply_transformations)

        # rotation 
        for axis, label in zip('XYZ', ['RotX', 'RotY', 'RotZ']):
            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, 360)
            slider.valueChanged.connect(lambda val, ax=axis: self._render_window.setRot(val, ax == 'X', ax == 'Y', ax == 'Z'))
            slider_layout.addWidget(QLabel(label))
            slider_layout.addWidget(slider)

        zoom_slider = QSlider(Qt.Horizontal)
        zoom_slider.setRange(10, 100)
        zoom_slider.setValue(50)  # Initial zoom level
        zoom_slider.valueChanged.connect(self._render_window.setZoom)
        slider_layout.addWidget(QLabel("Zoom"))
        slider_layout.addWidget(zoom_slider)

        slider_group.setLayout(slider_layout)
        controls_layout.addWidget(slider_group)

        layout.addWidget(container)
        layout.addLayout(controls_layout)
        layout.addWidget(translation_group)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())