from __future__ import print_function

import tempfile
import os
import subprocess
import tempfile
import meshio
import numpy as np
import voropy

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('GeomModel')

from petram.model import Model

from petram.namespace_mixin import NS_mixin
from petram.phys.vtable import VtableElement, Vtable, Vtable_mixin

debug = True

        
class GmshMeshActionBase(Model, Vtable_mixin, NS_mixin):
    hide_ns_menu = True
    has_2nd_panel = False
    isGmshMesh = True
    
    def __init__(self, *args, **kwargs):
        super(GmshMeshActionBase, self).__init__(*args, **kwargs)
        NS_mixin.__init__(self, *args, **kwargs)

    def attribute_set(self, v):
        v = super(GmshMeshActionBase, self).attribute_set(v)
        self.vt.attribute_set(v)
        return v

    def panel1_param(self):
        return self.vt.panel_param(self)
        
    def get_panel1_value(self):
        return self.vt.get_panel_value(self)

    def preprocess_params(self, engine):
        self.vt.preprocess_params(self)
        return

    def import_panel1_value(self, v):
        return self.vt.import_panel_value(self, v)

    def panel1_tip(self):
        return self.vt.panel_tip()

    def build_mesh(self, lines):
        raise NotImplementedError(
             "you must specify this method in subclass")
    
    
class GmshMesh(Model, NS_mixin):
    has_2nd_panel = False
    isMeshGroup = True        
    def __init__(self, *args, **kwargs):
        super(GmshMesh, self).__init__(*args, **kwargs)
        NS_mixin.__init__(self, *args, **kwargs)
        
    def attribute_set(self, v):
        v['geom_group'] = ''
        super(GmshMesh, self).attribute_set(v)
        return v
    
    def panel1_param(self):    
        return  [["Geometry", self.geom_group,  0, {},],
                 [None, None, 141, {"label": "Export...",
                                   "func": self.onExportMesh,
                                   "noexpand": True}],]
    
    def get_panel1_value(self):
        return (self.geom_group, None)
    
    def import_panel1_value(self, v):
        self.geom_group = str(v[0])        
    
    def get_possible_child(self):
        from .gmsh_mesh_actions import TransfiniteLine
        return [TransfiniteLine]
    
    def get_special_menu(self):
        return [('Build all', self.onBuildAll)]

    def onExportMesh(self, evt):
        print("export geom file")
        if not hasattr(self, "_txt_unrolled"):
            evt.Skip()
            return
        from ifigure.widgets.dialog import write
        parent = evt.GetEventObject()
        path = write(parent,
                     message = 'Enter .geo file name',
                     wildcard = '*.geo')
        if path != '':
            fid = open(path, 'w')
            fid.write('\n'.join(self._txt_unrolled))
            fid.close()
    
    def onBuildAll(self, evt):
        dlg = evt.GetEventObject()
        viewer = dlg.GetParent()
        engine = viewer.engine
        engine.build_ns()
        
        geom_root = self.root()['Geometry'][self.geom_group]
        if not hasattr(geom_root, "_txt_unrolled"):
            geom_root.onBuildAll(evt)

        self.build_mesh(geom_root)
        dlg.OnRefreshTree()
        evt.Skip()
        
    def build_mesh(self, geom_root, stop1=None, stop2=None):
        lines = [x.strip() for x in geom_root._txt_unrolled]
        num_entities = geom_root._num_entities
        
        children = [x for x in self.walk()]
        children = children[1:]

        from .gmsh_mesh_actions import MeshData
        meshdata = MeshData(lines, num_entities)
        
        for child in children:
            if not child.enabled: continue            
            child.vt.preprocess_params(child)
            if child is stop1: break            # for build before
            child.build_mesh(meshdata)
            if child is stop2: break            # for build after
            
        lines = meshdata.lines
        if debug:
            for l in lines:
                 print(l.strip())

        self._txt_unrolled = lines
        

        
