
import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('GeomModel')

from petram.model import Model
from petram.mfem_model import MFEM_GeomRoot # this is needed for backward compatibility

from petram.namespace_mixin import NS_mixin

class GeomBase(Model, NS_mixin):
    def __init__(self, *args, **kwargs):
        super(GeomBase, self).__init__(*args, **kwargs)
        NS_mixin.__init__(self, *args, **kwargs)
        
    def onItemSelChanged(self, evt):
        '''
        GUI response when model object is selected in
        the dlg_edit_model
        '''

        viewer = evt.GetEventObject().GetTopLevelParent().GetParent()
        viewer.set_view_mode('geom', self)
        
class GeomTopBase(GeomBase):
    def attribute_set(self, v):
        v = super(GeomBase, self).attribute_set(v)
        v['geom_finalized'] = False
        v['geom_timestamp'] = 0
        return v
