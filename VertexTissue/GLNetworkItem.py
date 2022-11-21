from OpenGL.GL import *
from OpenGL.GLUT import glutBitmapCharacter, glutInit, glutSwapBuffers, glutInitDisplayMode, GLUT_MULTISAMPLE
from OpenGL.GLUT.fonts import GLUT_BITMAP_9_BY_15
from pyqtgraph.opengl.GLGraphicsItem import GLGraphicsItem
from pyqtgraph.opengl.items.GLScatterPlotItem import GLScatterPlotItem
from pyqtgraph.Qt import QtCore, QtGui
from pyqtgraph.Qt.QtGui import QColor
from pyqtgraph import functions as fn

import numpy as np

__all__ = ['GLNetworkItem']

def glut_print_3D( xyz,  font,  text, r,  g , b , a):

    blending = False 
    if glIsEnabled(GL_BLEND) :
        blending = True

    #glEnable(GL_BLEND)
    glColor4f(r,g,b,a)
    glRasterPos3d(*xyz)
    for ch in text :
        glutBitmapCharacter( font , ctypes.c_int( ord(ch) ) )


    if not blending :
        glDisable(GL_BLEND) 

def glut_print_2D(  font,  text, r,  g , b , a):

    blending = False 
    if glIsEnabled(GL_BLEND) :
        blending = True


    glColor4f(r,g,b,a)
    

    glWindowPos2d(10.0, 10.0)
    for ch in text :
        glutBitmapCharacter( font , ctypes.c_int( ord(ch) ) )

    # glMatrixMode(GL_PROJECTION);
    # # glPopMatrix();
    # glMatrixMode(GL_MODELVIEW);
    # glPopMatrix();
    # glMatrixMode(GL_PROJECTION);
    # 
    # glMatrixMode(GL_MODELVIEW);
    
    


    if not blending :
        glDisable(GL_BLEND) 
    # glPopMatrix()

class GLNetworkItem(GLGraphicsItem):
    """A GLGraphItem displays graph information as
    a set of nodes connected by lines (as in 'graph theory', not 'graphics').
    Useful for drawing networks, trees, etc.
    """

    def __init__(self, draw_axes=False, msg='dummy', **kwds):
        GLGraphicsItem.__init__(self)

        self.edges = None
        self.edgeColor = QtGui.QColor(QtCore.Qt.GlobalColor.white)
        self.edgeWidth = 1.0

        self.nodeLabels=None
        self.draw_axes = draw_axes
        self.setData(**kwds)
        self.msg=msg

    def setData(self, **kwds):
        """
        Change the data displayed by the graph. 

        Parameters
        ----------
        edges : np.ndarray
            2D array of shape (M, 2) of connection data, each row contains
            indexes of two nodes that are connected.  Dtype must be integer
            or unsigned.
        edgeColor: QColor, array-like, optional.
            The color to draw edges. Accepts the same arguments as 
            :func:`~pyqtgraph.mkColor()`.  If None, no edges will be drawn.
            Default is (1.0, 1.0, 1.0, 0.5).
        edgeWidth: float, optional.
            Value specifying edge width.  Default is 1.0
        nodePositions : np.ndarray
            2D array of shape (N, 3), where each row represents the x, y, z
            coordinates for each node
        nodeColor : np.ndarray, QColor, str or array like
            2D array of shape (N, 4) of dtype float32, where each row represents
            the R, G, B, A vakues in range of 0-1, or for the same color for all
            nodes, provide either QColor type or input for 
            :func:`~pyqtgraph.mkColor()`
        nodeSize : np.ndarray, float or int
            Either 2D numpy array of shape (N, 1) where each row represents the
            size of each node, or if a scalar, apply the same size to all nodes
        **kwds
            All other keyword arguments are given to
            :meth:`GLScatterPlotItem.setData() <pyqtgraph.opengl.GLScatterPlotItem.setData>`
            to affect the appearance of nodes (pos, color, size, pxMode, etc.)
        
        Raises
        ------
        TypeError
            When dtype of edges dtype is not unisnged or integer dtype
        """

        if 'edges' in kwds:
            self.edges = kwds.pop('edges')
            if self.edges.dtype.kind not in 'iu':
                raise TypeError("edges array must have int or unsigned dtype.")

        if 'edgeColor' in kwds:
            edgeColor = kwds.pop('edgeColor')
            if edgeColor is not None:
                if type(edgeColor) != np.ndarray or len(edgeColor.shape)==1 or 1 in edgeColor.shape:
                    self.edgeColor = edgeColor
                else:
                    self.edgeColor = edgeColor

            else:
                self.edgeColor = None

        if 'edgeWidth' in kwds:
            self.edgeWidth = kwds.pop('edgeWidth')
        if 'nodePositions' in kwds:
            kwds['pos'] = kwds.pop('nodePositions')
        if 'nodeColor' in kwds:
            kwds['color'] = kwds.pop('nodeColor')
        if 'nodeSize' in kwds:
            kwds['size'] = kwds.pop('nodeSize')
        if 'nodeLabels' in kwds:
            self.nodeLabels = kwds.pop('nodeLabels')
        if 'msg' in kwds:
            self.msg=kwds['msg']

        self.pos=kwds['pos']



        self.update()

    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)
        
        glEnable( GL_BLEND );
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glutInit()
        glutInitDisplayMode(GLUT_MULTISAMPLE);


    def paint(self):
        if self.pos is None \
                or self.edges is None \
                or self.edgeColor is None:
            return None
        verts = self.pos
        edges = self.edges
        lbls = self.nodeLabels
        
        try:
            if self.draw_axes:
                glColor4f(0.5,0.5,0.5,1.0)
                glLineWidth(1.0)
                

                xhat=(1.0, 0.0, 0.0)
                yhat=(0.0, 1.0, 0.0)
                zhat=(0.0, 0.0, 1.0)
                origin=(0.0, 0.0, 0.0)



                glBegin(GL_LINES)

                glVertex3f(*origin)
                glVertex3f(*xhat)

                glVertex3f(*origin)
                glVertex3f(*yhat)



                glVertex3f(*origin)
                glVertex3f(*zhat)

                glEnd()

                glut_print_3D( xhat, GLUT_BITMAP_9_BY_15 , 'x' , 0.5 , .5 , .5 , 1.0 )
                glut_print_3D( yhat, GLUT_BITMAP_9_BY_15 , 'y' , 0.5 , .5 , .5 , 1.0 )
                glut_print_3D( zhat, GLUT_BITMAP_9_BY_15 , 'z' , 0.5 , .5 , .5 , 1.0 )




            glEnableClientState(GL_VERTEX_ARRAY)
            if lbls is not None:
                for pos, lbl  in zip(verts, lbls):
                    glut_print_3D( pos, GLUT_BITMAP_9_BY_15 , lbl , 0.5 , .5 , .5 , 1.0 )


            glVertexPointerf(verts)

            multicolor = not (len(self.edgeColor.shape)==1 or 1 in  self.edgeColor.shape)
            multiwidth = isinstance(self.edgeWidth, (list, tuple, np.ndarray))
            mode = GL_LINES


            glEnable(GL_LINE_SMOOTH)
            if not multiwidth:
                glLineWidth(self.edgeWidth)

            if not multicolor and not multiwidth:
                glColor4f(*self.edgeColor)
                for i, e in enumerate(edges):
                    glDrawElements(mode, 2, GL_UNSIGNED_INT, e)
            elif not multiwidth:
                for i, e in enumerate(edges):
                    glColor4f(*self.edgeColor[i])
                    glDrawElements(mode, 2, GL_UNSIGNED_INT, e)
            else:
                for i, e in enumerate(edges):
                    glColor4f(*self.edgeColor[i])
                    glLineWidth(self.edgeWidth[i])
                    glDrawElements(mode, 2, GL_UNSIGNED_INT, e)

            if self.msg:
                glut_print_2D(  GLUT_BITMAP_9_BY_15 , self.msg , 0.5 , .5 , .5 , 1.0 )


            # glutSwapBuffers()
        finally:
            glDisableClientState(GL_VERTEX_ARRAY)
        return None
