'''

    Model Tree to stroe MFEM model parameters

'''
from __future__ import print_function

import tempfile
import os
import subprocess
import traceback
import sys
import re
import time
from collections import defaultdict

import numpy as np
import warnings

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('GmshGeomModel')

from petram.model import Model
import time

import thread
from threading import Thread
try:
    from Queue import Queue, Empty
except ImportError:
    from queue import Queue, Empty  # python 3.x

import petram.geom.gmsh_config
from petram.geom.geom_model import GeomBase, GeomTopBase
from petram.namespace_mixin import NS_mixin
from petram.phys.vtable import VtableElement, Vtable, Vtable_mixin

debug = True
geom_key_dict = {'SurfaceBase': 'sb',
                 'PlaneSurface' : 'sp',
                 'Point': 'pt',
                 'Line': 'ln',
                 'Spline': 'sp'}


def get_gmsh_exe():
    macos_gmsh_location = '/Applications/Gmsh.app/Contents/MacOS/gmsh'
    if os.path.isfile(macos_gmsh_location):
        gmsh_executable = macos_gmsh_location
    else:
        gmsh_executable = 'gmsh'
    return gmsh_executable


def get_gmsh_major_version():
    gmsh_exe = get_gmsh_exe()
    try:
        out = subprocess.check_output(
                [gmsh_exe, '--version'],
                 stderr=subprocess.STDOUT
                 ).strip().decode('utf8')
    except:
        return -1
    ex = out.split('.')
    return int(ex[0])

use_gmsh_api = True
gmsh_Major=get_gmsh_major_version()
if gmsh_Major <= 3: use_gmsh_api = False

def enqueue_output(p, queue):
    while True:
        line = p.stdout.readline()
        
        queue.put(line.strip())
        if p.poll() is not None: 
            queue.put("End of Thread")
            return
        print(line.strip())
    queue.put("End of Thread")    
    
def collect_std_out(p,  verbose=True):
    q = Queue()
    t = Thread(target=enqueue_output, args=(p, q))
    t.daemon = True # thread dies with the program
    t.start()
    
    lines = []
    alive = True
    while True:

        time.sleep(0.01)
        
        try:  line = q.get_nowait() # or q.get(timeout=.1)
        except Empty:
            if p.poll() is not None:
                print('proces exited')
                break
            else:
                continue
        ec = p.poll()
        if ec is not None and ec < 0:
            print("RETURNIng due to this?")
            break  # on unix, this means process killed by a signal
        lines.append(line)
    return lines, p.poll()

def get_geom_key(obj):
    if obj.__class__ in geom_key_dict:
        return geom_key_dict[obj.__class__]
    name = obj.__class__.__name__
    key = ''.join([i.lower() for i in name if not i.isupper()])

    for k in geom_key_dict.keys():
        if geom_key_dict[k] == key:
            assert False, key + " is used for " + k.__name__
            
    geom_key_dict[obj.__class__] = key
    if debug: print(geom_key_dict)
    return key

class GeomObjs(dict):
    def duplicate(self):
        if not hasattr(self, "_past_keys"):
            self._past_keys = []            
        obj = GeomObjs(self)
        obj._past_keys = self._past_keys
        return obj
        
    def addobj(self, obj, name):
        key = ''.join([i for i in name if not i.isdigit()])
        if not hasattr(self, "_past_keys"):
            self._past_keys = []
        keys = self._past_keys
        nums = []
        for k in keys:
           t = ''.join([i for i in k if not i.isdigit()])
           if t == key:
              n = int(''.join([i for i in k if i.isdigit()]))
              nums.append(n)
        if len(nums) == 0:
           newkey = key+str(1)
        else:
           newkey = key+str(max(nums)+1)
        self[newkey] = obj
        self._past_keys.append(newkey)
        return newkey

class GmshPrimitiveBase(GeomBase, Vtable_mixin):
    hide_ns_menu = True
    has_2nd_panel = False
    isGeom = True
        
    def __init__(self, *args, **kwargs):
        super(GmshPrimitiveBase, self).__init__(*args, **kwargs)
        NS_mixin.__init__(self, *args, **kwargs)
        
    def attribute_set(self, v):
        v = super(GmshPrimitiveBase, self).attribute_set(v)
        self.vt.attribute_set(v)
        return v
    
    def panel1_param(self):
        from wx import BU_EXACTFIT
        b1 = {"label": "S", "func": self.onBuildBefore,
              "noexpand": True, "style": BU_EXACTFIT}
        b2 = {"label": "R", "func": self.onBuildAfter,
              "noexpand": True, "style": BU_EXACTFIT}
        
        ll = [[None, None, 241, {'buttons':[b1,b2],
                                 'alignright':True,
                                 'noexpand': True},],]
        ll.extend(self.vt.panel_param(self))
        return ll
        
    def get_panel1_value(self):
        return [None] + list(self.vt.get_panel_value(self))

    def preprocess_params(self, engine):
        self.vt.preprocess_params(self)
        return

    def import_panel1_value(self, v):
        return self.vt.import_panel_value(self, v[1:])

    def panel1_tip(self):
        return [None]+list(self.vt.panel_tip())

    #def build_geom(self, geom, objs):
    #    self._newobjs = []
    #    warnings.warn("Not implemented: " + self.__class__.__name__, Warning)        

    def gsize_hint(self, geom, objs):
        '''
        return quick estimate of geometry size min and max
        '''
        warnings.warn("Not implemented", Warning)
        return -1, -1
    
    def get_special_menu(self):
        return [('Build this step', self.onBuildAfter, None)]

    def _onBuildThis(self, evt, **kwargs):
        dlg = evt.GetEventObject().GetTopLevelParent()
        viewer = dlg.GetParent()
        engine = viewer.engine
        engine.build_ns()
        kwargs['gui_parent'] = dlg
        try:
            p  = self.parent
            if isinstance(p, GmshGeom):
                rootg = p
            else: # work plane
                rootg = p.parent
            rootg._geom_finalized = False
            
            od = os.getcwd()
            os.chdir(viewer.model.owndir())
            rootg.build_geom(**kwargs)
            os.chdir(od)            

        except:
            import ifigure.widgets.dialog as dialog               
            dialog.showtraceback(parent = dlg,
                                 txt='Failed to build geometry',
                                 title='Error',
                                 traceback=traceback.format_exc())
        dlg.OnRefreshTree()
        rootg.onUpdateGeoView(evt)

        
    def onBuildBefore(self, evt):
        dlg = evt.GetEventObject().GetTopLevelParent()
        mm = dlg.get_selected_mm()
        if mm is None: return

        self._onBuildThis(evt, stop1 = mm)
        evt.Skip()
        
    def onBuildAfter(self, evt):
        dlg = evt.GetEventObject().GetTopLevelParent()
        mm = dlg.get_selected_mm()
        if mm is None: return
        
        self._onBuildThis(evt, stop2 = mm)

        dlg = evt.GetEventObject().GetTopLevelParent()
        dlg.select_next_enabled()
        evt.Skip()

    def add_geom_sequence(self, geom):
        gui_name = self.fullname()
        gui_param = self.vt.make_value_or_expression(self)
        geom_name = self.__class__.__name__
        geom.add_sequence(gui_name, gui_param, geom_name)
        
class BrepFile(GeomTopBase):
    has_2nd_panel = False
    def __init__(self, *args, **kwargs):
        super(BrepFile, self).__init__(*args, **kwargs)
        NS_mixin.__init__(self, *args, **kwargs)
        
    @property
    def is_finalized(self):
        if not hasattr(self, "_geom_finalized"):
            self._geom_finalized = False
        return self._geom_finalized
    
    @property
    def _geom_brep(self):
        return self.brep_file_path

    @property
    def geom_finalized(self):
        if not hasattr(self, "_geom_finalized"):
            self._geom_finalized = False
        return self._geom_finalized
    
    @geom_finalized.setter
    def geom_finalized(self, value):
        self._geom_finalized = value
    
    def attribute_set(self, v):
        v = super(BrepFile, self).attribute_set(v)
        v['brep_file_path'] = ''
        v['geom_timestamp'] = 0
        v['geom_prev_algorithm'] = 2
        v['geom_prev_res'] = 30
        return v

    def panel1_param(self):
        import wx
        
        wc = "ANY|*|Brep|*.brep"
        return [["File(.brep)", None, 45, {'wildcard':wc}],        
                ["PreviewAlgorith", "Automatic", 4, {"style":wx.CB_READONLY,
                                                     "choices": ["Auto", "MeshAdpat",
                                                                 "Delaunay", "Frontal"]}],
                ["PreviewResolution", 30,  400, None],
                [None, None, 341, {"label": "Finalize Geom",
                                   "func": 'onBuildAll',
                                   "noexpand": True}],]

    def get_panel1_value(self):
        aname = {2: "Auto", 1: "MeshAdpat", 5: "Delaunay", 6:"Frontal"}
        txt = aname[self.geom_prev_algorithm]
        return [self.brep_file_path, txt, self.geom_prev_res, self]
       
    def import_panel1_value(self, v):
        aname = {2: "Auto", 1: "MeshAdpat", 5: "Delaunay", 6:"Frontal"}
        for k in aname:
            if v[1] == aname[k]:
                self.geom_prev_algorithm = k

        self.geom_prev_res = long(v[2])
        self.brep_file_path = str(v[0])
        
    def get_special_menu(self):
        return [('Load File', self.onBuildAll, None),]
    
    def onBuildAll(self, evt):
        '''
        filename : export geometry to a real file (for debug)
        '''
        import gmsh
        
        if not hasattr(self, "_gmsh4_data"):
            self._gmsh4_data = None
        if self._gmsh4_data is not  None:
            self._gmsh4_data[-1].finalize()
            
        objs = GeomObjs()
        self._objs = objs

        from petram.geom.gmsh_geom_wrapper import Geometry
        geom = Geometry()
        
        import gmsh
        geom.clear()        
        gmsh.model.occ.importShapes(self.brep_file_path, highestDimOnly=False)
        gmsh.model.occ.synchronize()

            
        # here we ask for 2D mesh for plotting.
        # size control is done based on geometry size.
        size = []
        for dim, tag in gmsh.model.getEntities():
            x1, y1, z1, x2, y2, z2 = gmsh.model.getBoundingBox(dim, tag)
            s = ((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)**0.5
            size.append((dim, tag, s))
        maxsize = max([x[-1] for x in size])
        lcar = defaultdict(lambda: np.inf)
        for dim, tag in gmsh.model.getEntities(1):
            x1, y1, z1, x2, y2, z2 = gmsh.model.getBoundingBox(dim, tag)
            s = ((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)**0.5
            bdimtags = gmsh.model.getBoundary(((dim, tag,),), oriented=False)
            for bdim, btag in bdimtags:
                lcar[btag] = min((lcar[btag], s/self.geom_prev_res))
        #ss = dict(lcar)
        #dim2_size = min([s[2] for s in ss if s[0]==2]+[3e20])
        #dim1_size = min([s[2] for s in ss if s[0]==1]+[3e20])

        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", maxsize/self.geom_prev_res)
        gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 1)
        geom.model.mesh.generate(1)
        
        gmsh.option.setNumber("Mesh.Algorithm", self.geom_prev_algorithm)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 1e22)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax",maxsize/5.)
        gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)        
        geom.model.mesh.generate(2)        


        self.geom_finalized = True
        self.geom_timestamp = time.ctime()        
        
        from petram.geom.read_gmsh import read_pts_groups, read_loops
        ptx, cells, cell_data = read_pts_groups(geom)
        l, s, v = read_loops(gmsh)
        self._gmsh4_data = (ptx, cells, cell_data, l, s, v, geom)
        
        ret = ptx, cells, {}, cell_data, {}

        # set clmax guess from geometry size
        clmax = maxsize/3.
        clmin = maxsize/300.
        self._clmax_guess = (clmax, clmin)
        self._geom_coords = ret

        dlg = evt.GetEventObject().GetTopLevelParent()
        viewer = dlg.GetParent()
        
        viewer.set_figure_data('geom', self.name(), ret)
        viewer.update_figure('geom', self.name())
        
        viewer._s_v_loop['geom'] = s, v
        viewer._s_v_loop['mesh'] = s, v

        evt.Skip()
        

class GmshGeom(GeomTopBase):
    has_2nd_panel = False
    def __init__(self, *args, **kwargs):
        super(GmshGeom, self).__init__(*args, **kwargs)
        NS_mixin.__init__(self, *args, **kwargs)
        
    @property
    def is_finalized(self):
        if not hasattr(self, "_geom_finalized"):
            self._geom_finalized = False
        return self._geom_finalized
    
    @property
    def geom_finalized(self):
        if not hasattr(self, "_geom_finalized"):
            self._geom_finalized = False
        return self._geom_finalized
    
    @geom_finalized.setter
    def geom_finalized(self, value):
        self._geom_finalized = value
    
    @property    
    def build_stop(self):
        if not hasattr(self, "_build_stop"):
            self._build_stop = (None, None)
        return self._build_stop
    
    def attribute_set(self, v):
        v = super(GmshGeom, self).attribute_set(v)
        v['geom_timestamp'] = 0
        v['geom_prev_algorithm'] = 2
        v['geom_prev_res'] = 30
        v['occ_parallel'] = False
        v['maxthreads'] = 1
        return v
        
    def get_possible_child(self):
        from .gmsh_primitives import Point, Line, Spline, Circle, Rect, Polygon, Box, Ball, Cone, Wedge, Cylinder, Torus, Extrude, Revolve, LineLoop, CreateLine, CreateSurface, CreateVolume, SurfaceLoop, Union, Intersection, Difference, Fragments, Copy, Remove, Move, Rotate, Flip, Scale, WorkPlane, CADImport, Fillet, Chamfer, Array, ArrayRot
        return [Point,  Line, Circle, Rect, Polygon, Spline, Box, Ball, Cone, Wedge, Cylinder, Torus, CreateLine, CreateSurface, CreateVolume, LineLoop, SurfaceLoop, Extrude, Revolve, Union, Intersection, Difference, Fragments, Copy, Remove, Move, Rotate, Flip, Scale, WorkPlane, CADImport, Fillet, Chamfer, Array, ArrayRot]
    
    def get_possible_child_menu(self):
        from .gmsh_primitives import Point, Line, Spline, Circle, Rect, Polygon, Box, Ball, Cone, Wedge, Cylinder, Torus, Extrude, Revolve, LineLoop, CreateLine, CreateSurface, CreateVolume, SurfaceLoop, Union, Intersection, Difference, Fragments, Copy, Remove, Move, Rotate, Flip, Scale, WorkPlane, CADImport, BrepImport, Fillet, Chamfer, Array, ArrayRot
        return [("", Point),("", Line), ("", Circle), ("", Rect), ("", Polygon),
                ("", Spline),("", Fillet), ("", Chamfer), 
                ("3D shape...", Box),
                ("", Ball), ("", Cone), ("", Wedge), ("", Cylinder),
                ("!", Torus),
                ("", CreateLine), ("", CreateSurface), ("", CreateVolume),
                ("", LineLoop), ("", SurfaceLoop),
                ("Protrude...", Extrude, "Extrude"), ("!", Revolve),
                ("", Copy), ("", Remove),
                ("Translate...", Move,), ("", Rotate),("", Flip),("", Scale),
                ("", Array), ("!", ArrayRot, "ArrayR"),
                ("Boolean...", Union),("",Intersection),("",Difference),("!",Fragments),
                ("", WorkPlane),
                ("Import...", BrepImport, "Brep"),("!", CADImport, "STEP/IGS"),
                ]
                
    def get_special_menu(self):
        if use_gmsh_api:
            return [('Build All', self.onBuildAll, None),
                    ('Export .brep', self.onExportBrep, None)]
        else:
            return [('Build All', self.onBuildAll, None),
                    ('Export .geo', self.onExportGeom, None)]
    
    def panel1_param(self):
        import wx
        return [["", "Geometry model using GMSH", 2, None],
                ["PreviewAlgorith", "Automatic", 4, {"style":wx.CB_READONLY,
                                                     "choices": ["Auto", "MeshAdpat",
                                                                 "Delaunay", "Frontal"]}],
                ["Preview Resolution", 30,  400, None],
                ["Preview #threads", self.maxthreads, 400, None],                
                [None, self.occ_parallel, 3, {"text":"OCC parallel boolean"}],
                [None, None, 341, {"label": "Finalize Geom",
                                   "func": 'onBuildAll',
                                   "noexpand": True}],]

    def get_panel1_value(self):
        aname = {2: "Auto", 1: "MeshAdpat", 5: "Delaunay", 6:"Frontal"}
        txt = aname[self.geom_prev_algorithm]
        return [None, txt, self.geom_prev_res, self.maxthreads, self.occ_parallel, self]
       
    def import_panel1_value(self, v):
        aname = {2: "Auto", 1: "MeshAdpat", 5: "Delaunay", 6:"Frontal"}
        for k in aname:
            if v[1] == aname[k]:
                self.geom_prev_algorithm = k

        self.geom_prev_res = long(v[2])
        self.maxthreads  =  long(v[3])
        self.occ_parallel  = v[4]

    def onBuildAll(self, evt):
        dlg = evt.GetEventObject().GetTopLevelParent()
        viewer = dlg.GetParent()
        engine = viewer.engine
        engine.build_ns()

        try:
            od = os.getcwd()
            os.chdir(viewer.model.owndir())
            self.build_geom(finalize = True, gui_parent=dlg)
            os.chdir(od)
        except:
            import ifigure.widgets.dialog as dialog               
            dialog.showtraceback(parent = dlg,
                                 txt='Failed to build geometry',
                                 title='Error',
                                 traceback=traceback.format_exc())
        dlg.OnRefreshTree()

        filename = os.path.join(viewer.model.owndir(), self.name())
        self.onUpdateGeoView(evt, filename = filename)

        '''
        if not use_gmsh_api:
            fid = open(filename + '.geo_unrolled', 'w')
            fid.write('\n'.join(self._txt_unrolled))
            fid.close()
        '''    
        self.geom_finalized = True
        self.geom_timestamp = time.ctime()
        evt.Skip()
    '''    
    def onUpdateGeoView3(self, evt, filename = None):
        dlg = evt.GetEventObject().GetTopLevelParent()
        viewer = dlg.GetParent()
        
        geo_text = self._txt_rolled[:]
        xyz = guess_geom_size(self._txt_unrolled)

        clmax = np.max(np.max(xyz, 0) - np.min(xyz, 0))/3.
        clmin = np.min(np.max(xyz, 0) - np.min(xyz, 0))/3.        
        self._clmax_guess = (clmax, clmin)
        geo_text.extend(['Show "*";',
                         'Mesh.CharacteristicLengthMax = '+str(clmax) + ';'])

        ret =  generate_mesh(geo_object = None,
                             dim = 2,
                             filename = filename,
                             num_quad_lloyd_steps=0,
                             num_lloyd_steps=0,                             
                             geo_text = geo_text)
        viewer.set_figure_data('geom', self.name(), ret)
        viewer.update_figure('geom', self.name())

        self._geom_coords = ret
        viewer._s_v_loop['geom'] = read_loops(self._txt_unrolled)
        viewer._s_v_loop['mesh'] = viewer._s_v_loop['geom']
    '''
    def onUpdateGeoView4(self, evt, filename = None):
        dlg = evt.GetEventObject().GetTopLevelParent()
        viewer = dlg.GetParent()
        ptx, cells, cell_data, l, s, v, geom = self._gmsh4_data
        ret = ptx, cells, {}, cell_data, {}

        # set clmax guess from geometry size
        xmin, ymin, zmin, xmax, ymax, zmax = geom.getBoundingBox()
        l = ((xmax-xmin)**2 + (ymax-ymin)**2 + (zmax-zmin)**2)**0.5
        clmax = l/3.
        clmin = l/300.
        self._clmax_guess = (clmax, clmin)
        
        self._geom_coords = ret
        viewer.set_figure_data('geom', self.name(), ret)
        viewer.update_figure('geom', self.name())
        
        viewer._s_v_loop['geom'] = s, v
        viewer._s_v_loop['mesh'] = s, v
        
    def onUpdateGeoView(self, evt, filename = None):       
        if globals()['gmsh_Major']==4 and use_gmsh_api:
            return self.onUpdateGeoView4(evt, filename = filename)
        else:
            assert False, "GMSH 3 is not supported"

    def walk_over_geom_chidlren(self, geom, stop1=None, stop2=None):
        self._build_stop = (None, None)
        
        children = [x for x in self.walk()]
        children = children[1:]
        for child in children:
            if hasattr(child, "_newobjs"): del child._newobjs
            
        children = self.get_children()
        for child in children:
            if not child.enabled: continue
            
            if len(child.get_children())==0:
                child.vt.preprocess_params(child)
                if child is stop1: break            # for build before
                child.add_geom_sequence(geom)
                if child is stop2: break            # for build after
                
            else:  # workplane
                children2 = child.get_children()
                child.vt.preprocess_params(child)
                if child is stop1: break            # for build before                

                do_break = False
                
                geom.add_sequence('WP_Start', 'WP_Start', 'WP_Start')
                for child2 in children2:
                    if not child2.enabled: continue                    
                    child2.vt.preprocess_params(child2)
                    if child2 is stop1:
                        do_break = True
                        break            # for build before
                    child2.add_geom_sequence(geom)                    
                    if child2 is stop2:
                        do_break = True                        
                        break            # for build after

                # translate 2D objects in 3D space
                #for x in org_keys: del objs2[x]
                child.add_geom_sequence(geom)
                geom.add_sequence('WP_End', 'WP_End', 'WP_End')
                
                if do_break: break
                if child is stop2: break            # for build after
        if stop1 is not None:
            self._build_stop = (stop1, None)
            return stop1.name()
        if stop2 is not None:
            self._build_stop = (None, stop2)
            return stop2.name()
        return self.name()

    def update_GUI_after_geom(self, data, objs):
        children = [x for x in self.walk()]
        children = children[1:]
        
        for child in children:
            if hasattr(child, "_newobjs"): del child._newobjs

        for child in children:
            if child.fullname() in data:
                dd = data[child.fullname()]
                child._objkeys = dd[0]
                child._newobjs = dd[1]  
                
        self._objs = objs        

                   
    def build_geom4(self, stop1=None, stop2=None, filename = None,
                    finalize = False, no_mesh=False, gui_parent=None):
        '''
        filename : export geometry to a real file (for debug)
        '''
        import gmsh
        
        if not hasattr(self, "_gmsh4_data"):
            self._gmsh4_data = None
        #if self._gmsh4_data is not  None:
        #    self._gmsh4_data[-1].finalize()

        from petram.geom.gmsh_geom_wrapper import Geometry
        geom = Geometry(PreviewResolution = self.geom_prev_res,
                        PreviewAlgorithm = self.geom_prev_algorithm,
                        OCCParallel = int(self.occ_parallel),
                        Maxthreads = self.maxthreads)
        
        geom.set_factory('OpenCASCADE')
        
        stopname = self.walk_over_geom_chidlren(geom, stop1=stop1, stop2=stop2)

        import wx
        if gui_parent is None:
            gui_parent = wx.GetApp().TopWindow

        L = len(geom.geom_sequence) + 3
        pgb = wx.ProgressDialog("Generating geometry...",
                                "", L, parent = gui_parent,
                                style = wx.PD_APP_MODAL|wx.PD_AUTO_HIDE|wx.PD_CAN_ABORT)
        def close_dlg(evt, dlg=pgb):
            pgb.Destroy()
        pgb.Bind(wx.EVT_CLOSE, close_dlg)
        
        gui_data, objs, brep_file, data = geom.run_generator(no_mesh = no_mesh,
                                                             finalize=finalize,
                                                             filename = stopname,
                                                             progressbar = pgb)
        pgb.Destroy()        
        self._geom_brep = brep_file
        self.update_GUI_after_geom(gui_data, objs)

        if data is None:  # if no_mesh = True
            return   
        # for the readablity I expend data here, do we need geom?        
        ptx, cells, cell_data, l, s, v = data
        self._gmsh4_data = (ptx, cells, cell_data, l, s, v, geom)

        return


    def build_geom(self, stop1=None, stop2=None, filename = None,
                   finalize = False, gui_parent=None):

        if globals()['gmsh_Major']==4 and use_gmsh_api:
            self.build_geom4(stop1=stop1, stop2=stop2,
                             filename=filename,
                             finalize=finalize,
                             gui_parent = gui_parent)
        else:
            assert False, "GMSH 3 is not supported"
            #self.build_geom3(stop1=stop1, stop2=stop2,
            #                 filename=filename,
            #                 finalize=finalize)

    def onExportGeom(self, evt):
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
            fid.write('\n'.join(self._txt_rolled))
            fid.close()
            
    def onExportBrep(self, evt):
        if not hasattr(self, "_geom_brep"):
            evt.Skip()
            return
        from ifigure.widgets.dialog import write
        parent = evt.GetEventObject()
        path = write(parent,
                     message = 'Enter .brep file name',
                     wildcard = '*.brep')
        if path != '':
            from shutil import copyfile
            copyfile(self._geom_brep, path)

    '''
    def load_gui_figure_data(self, viewer):
        import meshio
        filename = os.path.join(viewer.model.owndir(), self.name())
        msh_filename = filename + '.msh'
        if not os.path.exists(msh_filename):
            return 'geom', self.name(), None
        ret = meshio.read(msh_filename)

        filename = os.path.join(viewer.model.owndir(), self.name())
        filename = filename + '.geo_unrolled'
        if os.path.exists(filename):
            fid = open(filename, 'r')
            unrolled = [l.strip() for l in fid.readlines()]
            fid.close()
            viewer._s_v_loop['geom'] = read_loops(unrolled)        
            viewer._s_v_loop['mesh'] = viewer._s_v_loop['geom']

        return 'geom', self.name(), ret
    '''
    def is_viewmode_grouphead(self):
        return True
    

    
def check_dim(unrolled):
    for line in unrolled:
        if line.startswith('Volume'): return 3
    for line in unrolled:
        if line.find('Surface'): return 2
    return 1

def guess_geom_size(unrolled):
    points = []
    for line in unrolled:
        if line.startswith("Point("):
            try:
                coords = line.split("=")[1]
                coords = coords[coords.find("{")+1:coords.find("}")]
                xyz = np.array([float(x) for x in coords.split(",")[:3]])
                points.append(xyz)
            except:
                pass
    points = np.vstack(points)
    return points
            
def read_loops(unrolled):
    ll = {}  # line loop
    sl = {}  # surface loop
    v = {}
    s = {}

    def split_line(line):
        a, b = line.split("=")
        k = int(a[a.find("(")+1:a.find(")")])
        idx = [abs(int(x)) for x in b[b.find("{")+1:b.find("}")].split(",")]
        return k, idx
    
    for line in unrolled:
        line = line.strip()
        if line.startswith("Surface Loop("):
            k, idx = split_line(line)
            sl[k] = idx
        elif line.startswith("Line Loop("):
            k, idx = split_line(line)
            ll[k] = idx
        elif line.startswith("Volume("):
            k, idx = split_line(line)
            v[k] = idx
        elif line.startswith("Plane Surface("):
            k, idx = split_line(line)
            s[k] = idx
        elif line.startswith("Surface("):
            k, idx = split_line(line)
            s[k] = idx
        else:
            pass

    for kv in v.keys():
        tmp = []
        for k in v[kv]:
            tmp.extend(sl[k])
        v[kv] = list(set(tmp))
    for ks in s.keys():
        tmp = []
        for k in s[ks]:
            tmp.extend(ll[k])
        s[ks] = list(set(tmp))
    return s, v

'''
def BoolFramgents_extra(name, shape_type,  inputs, delete = True):
    txt = '{}[] = {}{{{} {{{}}}; {}}} {{{} {{{}}}; {}}};'.format(
                name, 
                'BooleanFragments',
                shape_type,
                ','.join(str(e) for e in inputs[:1]),
                'Delete;' if delete else '',
                shape_type,
                ','.join(str(e) for e in inputs[1:]),
                'Delete;' if delete else '')
    return txt

def generate_mesh(
        geo_object = None,
        optimize=True,
        num_quad_lloyd_steps=10,
        num_lloyd_steps=1000,
        verbose=True,
        dim=3,
        prune_vertices=True,
        filename=None,
        geo_text=None,
        extra=None,
        bin = '-bin',
        verbosity = '4'
        ):
    from pygmsh.helper import _get_gmsh_exe, _is_flat
    import meshio
    import voropy    

    if filename is None:
        handle, geo_filename = tempfile.mkstemp(suffix='.geo')
    else:
        geo_filename = filename + '.geo'
        handle = os.open(geo_filename,  os.O_WRONLY | os.O_CREAT |
                         os.O_TRUNC)
    extra = [] if extra is None else extra    

    if geo_object is not None:
       rolled =  geo_object.get_code().encode().split("\n")
    elif geo_text is not None:
       rolled = geo_text

    if dim == 0:
        rolled.extend(extra)
        rolled.append("Geometry.OldNewReg=0;\n")
        rolled.append('Printf("Number of entitites, : %g, %g, %g, %g :", newp, newl, news, newv);\n')
    os.write(handle, "\n".join(rolled))
    os.close(handle)

    gmsh_executable = _get_gmsh_exe()

    if dim > 0:
        if filename is None:
            handle, msh_filename = tempfile.mkstemp(suffix='.msh')
            os.close(handle)            
        else:
            msh_filename = filename + '.msh'
        cmd = [
            gmsh_executable, '-v', verbosity, 
            '-{}'.format(dim), bin, geo_filename, '-o', msh_filename
            ]
        if num_quad_lloyd_steps > 0:
            cmd += ['-optimize_lloyd', str(num_quad_lloyd_steps)]
    elif dim < 0:
        if filename is None:
            handle, msh_filename = tempfile.mkstemp(suffix='.msh')
            os.close(handle)            
        else:
            msh_filename = filename + '.msh'
        cmd = [
            gmsh_executable, bin, geo_filename, '-o', msh_filename
            ]
    else:
        if filename is None:
            handle, geou_filename = tempfile.mkstemp(suffix='.geo_unrolled')
            os.close(handle)         
        else:
            geou_filename = filename + '.geo_unrolled'
        cmd = [
            gmsh_executable,
            '-{}'.format(dim), geo_filename, '-o', geou_filename
            ]

    if globals()['gmsh_Major']==4:
        cmd.extend(['-format', 'msh2'])
    print("calling gmsh", cmd)
    cmd = [x for x in cmd if x != '']
    # http://stackoverflow.com/a/803421/353337
    p = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        bufsize = 0)
    
    stdout = collect_std_out(p, True)
    stdoutdata = stdout[0]
    print("exit code", stdout[1])
    
#    for line in stdout[0]:
#        if verbose:
#            print(line.decode('utf-8'), end='')
#            sys.stdout.flush()
#        stdoutdata.append(line)

    ansi_escape = re.compile(r'(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]')

    if dim == 0:
        entity_relations = {'Volume':{}, 'Surface':{}, 'Line':{}}
        for x in stdoutdata:
            x = ansi_escape.sub('', x)                
            if x.startswith("Number of entitites,"):
                txt = x[x.find("Number of entitites,"):].split(":")
                num_entities = [int(x)-1 for x in txt[1].split(',')]
            if x.startswith("Boundary(Volume{"):
                id = int(x[x.find("{")+1:x.find("}")])
                nums = [int(o) for o in x.split("=")[-1].split(" ") if len(o) > 0]
                entity_relations['Volume'][id] = nums
            if x.startswith("Boundary(Surface{"):                
                id = int(x[x.find("{")+1:x.find("}")])
                nums = [int(o) for o in x.split("=")[-1].split(" ") if len(o) > 0]
                entity_relations['Surface'][id] = nums                           
            if x.startswith("PointsOf(Line{"):                                
                id = int(x[x.find("{")+1:x.find("}")])
                nums = [int(o) for o in x.split("=")[-1].split(" ") if len(o) > 0 ]
                entity_relations['Line'][id] = nums
                           
        fid = open(geou_filename, 'r')
        lines = fid.readlines()
        fid.close()
        #if filename is None:            
        #    os.remove(geo_filename)
        #    os.remove(geou_filename)
        print(entity_relations)
        return lines, rolled, (num_entities, entity_relations)

    assert stdout[1] == 0,\
        'Gmsh exited with error (return code {}).'.format(p.returncode)

    # meshio does not read $Periodic....
    fid = open(msh_filename, 'r')
    lines = fid.readlines()
    fid.close()
    has_periodic = False
    for n1, l in enumerate(lines):
        if l.strip() == '$Periodic':
            has_periodic = True
            ps = n1
            break
    if has_periodic:
        for n2, l in enumerate(lines):
            if l.strip() == '$EndPeriodic':
                pe = n2
                break
        print(n1, n2)
        lines = lines[:ps]+lines[pe+1:]
        fid = open(msh_filename, 'w')
        fid.write(''.join(lines))
        fid.close()

    X, cells, pt_data, cell_data, field_data = meshio.read(msh_filename)

    # clean up
    if filename is None:    
        #os.remove(geo_filename)
        #os.remove(msh_filename)
        pass

    # Lloyd smoothing
    if ('triangle' not in cells) or (not _is_flat(X)):
        if verbose:
            print(
                'Not performing Lloyd smoothing '
                '(only works for flat triangular meshes).'
                )
        return X, cells, pt_data, cell_data, field_data
    if dim < 0:
        return X, cells, pt_data, cell_data, field_data
    
    if num_lloyd_steps == 0 and num_quad_lloyd_steps == 0:
        return X, cells, pt_data, cell_data, field_data
    
    if verbose:
        print('Lloyd smoothing...')
    # find submeshes
    a = cell_data['triangle']['geometrical']
    # http://stackoverflow.com/q/42740483/353337
    submesh_bools = {v: v == a for v in np.unique(a)}

    X, cells['triangle'] = voropy.smoothing.lloyd_submesh(
            X, cells['triangle'], submesh_bools,
            tol=0.0, max_steps=num_lloyd_steps,
            verbose=False
            )

    if prune_vertices:
        # Make sure to include only those vertices which belong to a triangle.
        uvertices, uidx = np.unique(cells['triangle'], return_inverse=True)
        cells = {'triangle': uidx.reshape(cells['triangle'].shape)}
        cell_data = {'triangle': cell_data['triangle']}
        X = X[uvertices]
        for key in pt_data:
            pt_data[key] = pt_data[key][uvertices]

    return X, cells, pt_data, cell_data, field_data
'''    
