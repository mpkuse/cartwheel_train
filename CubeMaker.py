# Borrowed from : https://www.panda3d.org/forums/viewtopic.php?t=4879
from panda3d.core import *


class CubeMaker:
    def __init__(self):
        # self.smooth = True/False
        # self.uv = True/False or Spherical/Box/...
        # self.color = Method1/Method2/...
        # self.subdivide = 0/1/2/...
        self.size = 1.0

    def generate(self):
        format = GeomVertexFormat.getV3()
        data = GeomVertexData("Data", format, Geom.UHStatic)
        vertices = GeomVertexWriter(data, "vertex")

        size = self.size
        vertices.addData3f(-size, -size, -size)
        vertices.addData3f(+size, -size, -size)
        vertices.addData3f(-size, +size, -size)
        vertices.addData3f(+size, +size, -size)
        vertices.addData3f(-size, -size, +size)
        vertices.addData3f(+size, -size, +size)
        vertices.addData3f(-size, +size, +size)
        vertices.addData3f(+size, +size, +size)

        triangles = GeomTriangles(Geom.UHStatic)

        def addQuad(v0, v1, v2, v3):
            triangles.addVertices(v0, v1, v2)
            triangles.addVertices(v0, v2, v3)
            triangles.closePrimitive()

        addQuad(4, 5, 7, 6) # Z+
        addQuad(0, 2, 3, 1) # Z-
        addQuad(3, 7, 5, 1) # X+
        addQuad(4, 6, 2, 0) # X-
        addQuad(2, 6, 7, 3) # Y+
        addQuad(0, 1, 5, 4) # Y+

        geom = Geom(data)
        geom.addPrimitive(triangles)

        node = GeomNode("CubeMaker")
        node.addGeom(geom)

        return NodePath(node)


class SphereMaker:
    def __init__(self):
        a= 2


    def generate(self, radius=1 ):
        sph = loader.loadModel( 'cg_models/sphere' )

        nodePath = NodePath( sph )
        nodePath.setTransparency(TransparencyAttrib.MAlpha)
        nodePath.setAlphaScale(0.5)

        return nodePath
