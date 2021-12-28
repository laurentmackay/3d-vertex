from OpenGL.GL import *
from pyqtgraph.opengl.GLGraphicsItem import GLGraphicsItem
from pyqtgraph.opengl.items.GLScatterPlotItem import GLScatterPlotItem
from pyqtgraph.Qt import QtCore, QtGui
from pyqtgraph.Qt.QtGui import QColor
from pyqtgraph import functions as fn

import numpy as np

__all__ = ['GLNetworkItem']

class GLNetworkItem(GLGraphicsItem):
    """A GLGraphItem displays graph information as
    a set of nodes connected by lines (as in 'graph theory', not 'graphics').
    Useful for drawing networks, trees, etc.
    """

    def __init__(self, **kwds):
        GLGraphicsItem.__init__(self)

        self.edges = None
        self.edgeColor = QtGui.QColor(QtCore.Qt.GlobalColor.white)
        self.edgeWidth = 1.0

        # self.scatter = GLScatterPlotItem()
        # self.scatter.setParentItem(self)
        self.setData(**kwds)

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
        self.pos=kwds['pos']
        self.update()

    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)


    def paint(self):
        if self.pos is None \
                or self.edges is None \
                or self.edgeColor is None:
            return None
        verts = self.pos
        edges = self.edges
        glEnableClientState(GL_VERTEX_ARRAY)
        try:
            glVertexPointerf(verts)

            multicolor = not (len(self.edgeColor.shape)==1 or 1 in  self.edgeColor.shape)
            multiwidth = isinstance(self.edgeWidth, (list, tuple, np.ndarray))
            mode = GL_LINE_STRIP
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
        finally:
            glDisableClientState(GL_VERTEX_ARRAY)
        return None
