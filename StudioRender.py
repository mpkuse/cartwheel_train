""" Renderer for Game models
        The renders will have a step() function which will return a set of <q,P,N>.
        Need some mechanism for virtual-camera-pose monte-carlo to avoid
        no-go-zones.

        Created : 22th Feb, 2017
        Author  : Manohar Kuse <mpkuse@connect.ust.hk>
"""

from direct.showbase.ShowBase import ShowBase
import code

class StudioRender(ShowBase):

    def __init__(self):
        ShowBase.__init__(self)

        self.scene = self.loader.loadModel( 'all_models/Bundes/Bundes.obj' )
        self.scene.reparentTo(self.render)

        self.scene.setScale(0.01)

        print self.scene.getChildren()
        code.interact( local=locals() )
