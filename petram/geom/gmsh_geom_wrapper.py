
from __future__ import print_function

import numpy as np
import time
import tempfile
import multiprocessing as mp
from Queue import Empty as QueueEmpty

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('GmshGeomWrapper')

import petram.geom.gmsh_config as gmsh_config
import gmsh

from petram.phys.vtable import VtableElement, Vtable
from petram.geom.gmsh_geom_model import GmshPrimitiveBase as GeomPB
from petram.geom.gmsh_geom_model import get_geom_key

class Polygon(object):
    def __init__(self, s, ll, lcar):
        self.surface = SurfaceID(s)
        self.line_loop = ll
        self.lcar = lcar

class GeomIDBase(int):
   def __repr__(self):
       return self.__class__.__name__+"("+str(int(self))+")"

class VertexID(GeomIDBase):
   def __add__(self, v):
       return VertexID(int(self) + v)

class LineID(GeomIDBase):
   def __add__(self, v):
       return LineID(int(self) + v)
   def __neg__(self):
       return LineID(-int(self))
    
class SurfaceID(GeomIDBase):
   def __add__(self, v):
       return SurfaceID(int(self) + v)
   def __neg__(self):
       return SurfaceID(-int(self))
   
class VolumeID(GeomIDBase):   
   def __add__(self, v):
       return VolumeID(int(self) + v)
   def __neg__(self):
       return VolumeID(-int(self))
    
class LineLoopID(GeomIDBase):   
   def __add__(self, v):
       return LineLoopID(int(self) + v)
   def __neg__(self):
       return LineLoopID(-int(self))
    
class SurfaceLoopID(GeomIDBase):   
   def __add__(self, v):
       return SurfaceLoopID(int(self) + v)
   def __neg__(self):
       return SurfaceLoopID(-int(self))

def id2dimtag(en):
    if isinstance(en, VertexID):
        return (0, int(en))
    elif isinstance(en, LineID):
        return (1, int(en))
    elif isinstance(en, SurfaceID):
        return (2, int(en))
    elif hasattr(en, 'surface'):
        return (2, int(en))
    elif isinstance(en, VolumeID):
        return (3, int(en))
    else:
        assert False, "Illegal entity"+str(en)

def get_dimtag(entity):
    dimtags = []
    for en in entity:
        dimtags.append(id2dimtag(en))
    return dimtags

def dimtag2id(dimtags):        
    out3 = []; out2 = []; out1 = []; out0 = []
    for dim, tag in dimtags:
        if dim == 3 and not tag in out3:
           out3.append(VolumeID(tag))                               
        elif dim == 2 and not tag in out2:
           out2.append(SurfaceID(tag))                               
        elif dim == 1 and not tag in out1:
           out1.append(LineID(tag))
        elif dim == 0 and not tag in out1:
           out0.append(VertexID(tag))
    return out3 + out2 + out1 + out0

def get_target1(objs, targets, cls):
    # this is when target type is given
    if cls == 'l': cc = LineID
    if cls == 'v': cc = VolumeID
    if cls == 'f': cc = SurfaceID
    if cls == 'p': cc = VertexID    
    
    return [objs[t] if t in objs else cc(t)  for t in targets]
  
def get_target2(objs, targets):
    # this is when target type is given
    ret = []
    for t in targets:
        if t in objs:
           ret.append(objs[t])
        else:
           if t.startswith("p"): ret.append(VertexID(int(t[1:])))
           if t.startswith("l"): ret.append(LineID(int(t[1:])))
           if t.startswith("f"): ret.append(SurfaceID(int(t[1:])))
           if t.startswith("v"): ret.append(VolumeID(int(t[1:])))         
    return ret 

class Geometry(object):
    def __init__(self, *args, **kwargs):
        self._point_loc = {}

        self.geom_prev_res = kwargs.pop('PreviewResolutio', 30)
        self.geom_prev_algorithm = kwargs.pop('PreviewAlgorithm', 2) 
        self.occ_parallel = kwargs.pop('OCCParallel', 0)
        self.maxthreads = kwargs.pop('Maxthreads', 1)
        
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.option.setNumber("Geometry.OCCParallel", self.occ_parallel)        
        modelname = kwargs.pop("modelname", "model1")
        gmsh.clear()
        gmsh.model.add(modelname)

        self.model = gmsh.model
        self.factory = gmsh.model.occ

        #self.p = VertexID(0) 
        #self.l = LineID(0)   
        #self.ll = LineLoopID(0)  
        #self.s = SurfaceID(0)   
        #self.sl = SurfaceLoopID(0)
        #self.v = VolumeID(0)      
        self.geom_sequence = []
        
        self._point = {}
        self._point_mask = []
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)

        write_log = kwargs.pop('write_log', False)
        if write_log:
            self.logfile = tempfile.NamedTemporaryFile('w', delete = True)
        else:
            self.logfile = None
        self.queue = kwargs.pop("queue", None)
            
    def set_factory(self, factory_type):
        pass

    def clear(self):
        gmsh.clear()
        
    def getBoundingBox(self):
        xmax = -np.inf
        xmin =  np.inf
        ymax = -np.inf
        ymin =  np.inf
        zmax = -np.inf
        zmin =  np.inf
        
        def update_maxmin(dim, tag, xmin, ymin, zmin, xmax, ymax, zmax):
            x1, y1, z1, x2, y2, z2 = self.model.getBoundingBox(dim, tag)           
            xmax = np.max([xmax, x2])
            ymax = np.max([ymax, y2])
            zmax = np.max([zmax, z2])
            xmin = np.min([xmin, x1])
            ymin = np.min([ymin, y1])
            zmin = np.min([zmin, z1])
            return xmin, ymin, zmin, xmax, ymax, zmax
         
        #if (self.model.getEntities(3)) != 0:
        for dim, tag in self.model.getEntities():
            xmin, ymin, zmin, xmax, ymax, zmax = update_maxmin(dim, tag,
                                                               xmin, ymin, zmin,
                                                               xmax, ymax, zmax)
        '''      
        elif len(self.model.getEntities(2)) != 0:
           for dim, tag in self.model.getEntities(2):
              xmin, ymin, zmin, xmax, ymax, zmax = update_maxmin(dim, tag,
                                                                 xmin, ymin, zmin,
                                                                 xmax, ymax, zmax)
        elif len(self.model.getEntities(1)) != 0:
           for dim, tag in self.model.getEntities(1):
              xmin, ymin, zmin, xmax, ymax, zmax = update_maxmin(dim, tag,
                                                                 xmin, ymin, zmin,
                                                                 xmax, ymax, zmax)
        else:
           for dim, tag in self.model.getEntities(0):
              xmin, ymin, zmin, xmax, ymax, zmax = update_maxmin(dim, tag,
                                                                 xmin, ymin, zmin,
                                                                 xmax, ymax, zmax)
        '''
        return xmin, ymin, zmin, xmax, ymax, zmax
     
    def getObjSizes(self):
        size = []
        for dim, tag in self.model.getEntities():
            x1, y1, z1, x2, y2, z2 = self.model.getBoundingBox(dim, tag)
            s = ((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)**0.5
            size.append((dim, tag, s))
        return size
     
    def getVertexCL(self):
        from collections import defaultdict
        
        lcar = defaultdict(lambda: np.inf)
        
        for dim, tag in self.model.getEntities(1):
            x1, y1, z1, x2, y2, z2 = self.model.getBoundingBox(dim, tag)
            s = ((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)**0.5
            bdimtags = self.model.getBoundary(((dim, tag,),), oriented=False)
            for bdim, btag in bdimtags:
                lcar[btag] = min((lcar[btag], s))
        return dict(lcar)

    
    def getEntityNumberingInfo(self):
        '''
        Numbering info is collected to understand the change of dimtags
        after brep is saved/loaded. This info may be useful for caching
        the geometry in future?
        '''
        info_digits = 7
        from collections import defaultdict
        
        #self.factory.removeAllDuplicates()
        self.factory.synchronize()        
        
        points = [];
        edges = [];
        faces = [];
        volumes = [];
        for dimtag in self.model.getEntities(3):
            volumes.append(dimtag[1])
            f = self.model.getBoundary([dimtag], combined = False, oriented = False)
            for dim, tag in f:
                if not tag in faces: faces.append(tag)
                e = self.model.getBoundary([(dim,tag)], combined = False, oriented = False)
                for dim, tag in e:
                    if not tag in edges: edges.append(tag)
                    p = self.model.getBoundary([(dim,tag)], combined = False, oriented = False)
                    for dim, tag in p:
                        if not tag in points: points.append(tag)
        for dimtag in self.model.getEntities(2):
            dim, tag = dimtag
            if not tag in faces: faces.append(tag)
            e = self.model.getBoundary([(dim,tag)], combined = False, oriented = False)
            for dim, tag in e:
                if not tag in edges: edges.append(tag)
                p = self.model.getBoundary([(dim,tag)], combined = False, oriented = False)
                for dim, tag in p:
                    if not tag in points: points.append(tag)
        for dimtag in self.model.getEntities(1):
            dim, tag = dimtag            
            if not tag in edges: edges.append(tag)
            p = self.model.getBoundary([(dim,tag)], combined = False, oriented = False)
            for dim, tag in p:
                if not tag in points: points.append(tag)
        for dimtag in self.model.getEntities(0):
            dim, tag = dimtag 
            if not tag in points: points.append(tag)           

        map = points, edges, faces, volumes
        print("entities", points, edges, faces, volumes)        
        return map
        

    def applyEntityNumberingInfo(self, map, objs):
        map1  = self.getEntityNumberingInfo()
        point_map = dict(zip(map[0], map1[0]))
        edge_map = dict(zip(map[1], map1[1]))
        face_map = dict(zip(map[2], map1[2]))
        volume_map = dict(zip(map[3], map1[3]))        
        for key in objs:
            if isinstance(objs[key], VertexID):
                objs[key] = VertexID(point_map[objs[key]])
            if isinstance(objs[key], LineID):
                objs[key] = LineID(edge_map[objs[key]])
            if isinstance(objs[key], SurfaceID):
                objs[key] = SurfaceID(face_map[objs[key]])
            if isinstance(objs[key], VolumeID):
                objs[key] = VolumeID(volume_map[objs[key]])
       
    @staticmethod
    def write(filename):
        gmsh.write(filename)
        
    @staticmethod        
    def finalize():
        gmsh.finalize()

    @property
    def dim(self):
        if len(self.model.getEntities(3)) > 0: return 3
        if len(self.model.getEntities(2)) > 0: return 2
        if len(self.model.getEntities(1)) > 0: return 1
        return 0
     
    def add_point(self, p, lcar=0.0, mask=True):
        p = tuple(p)
        #if not p in self._point_loc:
        pp = self.factory.addPoint(p[0], p[1], p[2], lcar)
        self._point_loc[p] = VertexID(pp)
        #print("made point ", pp, p)
            
        p_id = self._point_loc[p]
        self._point[p_id]=np.array(p)
        if mask : self._point_mask.append(p_id)
        return p_id
        
    def add_line(self, p1, p2):
        l = self.factory.addLine(p1, p2)
        return LineID(l)
      
    def add_circle_arc(self, p2, pc, p3):
        l = self.factory.addCircleArc(p2, pc, p3)
        return LineID(l)        

    def add_spline(self, pts, remove_control=True):
        l = self.factory.addSpline(pts)
        if remove_control:
           dimtags = [(0, x) for x in pts[1:-1]]
           self.factory.remove(dimtags)
        return LineID(l)

    def add_plane_surface(self, tags):
        # tags : 1st element exterier, others makes hole
        tags = list(np.atleast_1d(tags))
        #self.factory.synchronize()                        
        s = self.factory.addPlaneSurface(tags)
        #self.factory.synchronize()                                
        return SurfaceID(s)

    def add_surface_filling(self, tags):
        tags = list(np.atleast_1d(tags))
        print("calling wire")
        wire = self.factory.addWire(tags)
        print("calling filling", wire)        
        s = self.factory.addSurfaceFilling(wire)
        return SurfaceID(s)
       
    def add_line_loop(self, pts, sign=None):
        tags = list(np.atleast_1d(pts))
        if sign is not None:
           for k, v in enumerate(sign):
               if not v: tags[k] = -tags[k]
              
        #self.factory.synchronize()                                
        #en1 = self.model.getEntities(1)
        
        ll = self.factory.addWire(tags, checkClosed=True)

        #self.factory.synchronize()                                
        #en2 = self.model.getEntities(1)
        #if len(en1) != len(en2):
        #  print("removing", tags[-1])
        #  self.factory.remove(((1, abs(tags[-1])),))

        # (note)
        #   somehow, addWire create a duplicated line sometimes
        #   here I delete input lines to enforce re-numbering.
        #
        dimtags = [(1, x) for x in tags]
        self.factory.remove(dimtags)
           
        #self.factory.synchronize()
        #en3 = self.model.getEntities(1)
        #print(en1, en2, en3)
        return LineLoopID(ll)
    
    def add_curve_loop(self, pts, sign=None):
        tags = list(np.atleast_1d(pts))
        if sign is not None:
           for k, v in enumerate(sign):
               if not v: tags[k] = -tags[k]
               
        ll = self.factory.addCurveLoop(tags)
        return LineLoopID(ll)
        
    def add_surface_loop(self, sl):
        tags = list(np.atleast_1d(sl))

        sl = self.factory.addSurfaceLoop(tags)
        return SurfaceLoopID(sl)

    def add_sphere(self, x, y, z, radius):
        v = self.factory.addSphere(x, y, z, radius)
        return VolumeID(v)
     
    def add_cone(self, x, y, z, dx, dy, dz, r1, r2, angle):
        v = self.factory.addCone(x, y, z, dx, dy, dz, r1, r2, angle=angle)
        return VolumeID(v)
   
    def add_wedge(self, x, y, z, dx, dy, dz, ltx):
        v = self.factory.addWedge(x, y, z, dx, dy, dz, -1, ltx)
        return VolumeID(v)

    def add_cylinder(self, x, y, z, dx, dy, dz, r, angle):
        v = self.factory.addCylinder(x, y, z, dx, dy, dz, r, angle=angle)
        return VolumeID(v)
     
    def add_torus(self, x, y, z, r1, r2, angle):
        v = self.factory.addTorus(x, y, z, r1, r2, -1, angle)
        return VolumeID(v)
        
    def add_volume(self, shells):
        tags = list(np.atleast_1d(shells))              
        v = self.factory.addVolume(tags)
        return VolumeID(v)
     
    def add_ellipse_arc(self, startTag, centerTag, endTag):
        a =  self._point[startTag] - self._point[centerTag]
        b =  self._point[endTag] - self._point[centerTag]
        if np.sum(a*a) > np.sum(b*b):
            l = self.factory.addEllipseArc(startTag, centerTag, endTag)
        else:
            l = self.factory.addEllipseArc(endTag, centerTag, startTag)
        return LineID(l)
                      
    def add_polygon(self, pos, lcar = 0.0):
        pts = [self.add_point(p, lcar=lcar) for p in pos]
        lns = [self.add_line(pts[i], pts[i+1]) for i in range(len(pts)-1)]
        lns.append(self.add_line(pts[-1], pts[0]))
        ll = self.add_line_loop(lns)
        sl = self.add_plane_surface((ll,))
        ret =  Polygon(sl, ll, lcar)
        return ret

    def fillet(self, volumes, curves, radii, removeVolume=True):
        volumeTags = list(np.atleast_1d(volumes))
        curveTags = list(np.atleast_1d(curves))
        radii = list(np.atleast_1d(radii))
        outTags = self.factory.fillet(volumeTags,
                                curveTags,
                                radii,
                                removeVolume=removeVolume)
        return [VolumeID(v[1]) for v in outTags]        

    def chamfer(self, volumes, curves, surfaces, distances, removeVolume=True):
        volumeTags = list(np.atleast_1d(volumes))
        curveTags = list(np.atleast_1d(curves))
        surfaceTags = list(np.atleast_1d(surfaces))        
        distances = list(np.atleast_1d(distances))
        print("surfaceTags", surfaceTags)
        outTags = self.factory.chamfer(volumeTags,
                                       curveTags,
                                       surfaceTags,
                                       distances,
                                       removeVolume=removeVolume)
        return [VolumeID(v[1]) for v in outTags]        
     
    def import_shapes(self, fileName, highestDimOnly=True, format=""):
        out_dimtags = self.factory.importShapes(fileName,
                                                highestDimOnly=highestDimOnly,
                                                format="")
        return dimtag2id(out_dimtags)
     
    def _boolean_xxx(self, m, input_entity, tool_entity,
                     removeObject=False, removeTool=False,
                     delete=False):
       
        dimtag1 = get_dimtag(input_entity)
        dimtag2 = get_dimtag(tool_entity)
                               
        if delete:
             removeObject=True
             removeTool=True

        m = getattr(self.factory, m)
        dimtag3, dimtagMap = m(dimtag1, dimtag2,
                               removeObject=removeObject,
                               removeTool=removeTool)
        self.factory.synchronize()
        self.model.getEntities()
        return dimtag2id(dimtag3)                
        
    def boolean_intersection(self, input_entity, tool_entity,
                             removeObject=False, removeTool=False, delete=False):
        return self._boolean_xxx('intersect', input_entity, tool_entity,
                                 removeObject=removeObject, removeTool=removeTool,
                                 delete=delete)
                               
    def boolean_union(self, input_entity, tool_entity,
                      removeObject=False, removeTool=False, delete=False):
        return self._boolean_xxx('fuse', input_entity, tool_entity,
                                 removeObject=removeObject, removeTool=removeTool,
                                 delete=delete)

     
    def boolean_difference(self, input_entity, tool_entity,
                           removeObject=False, removeTool=False, delete=False):
        return self._boolean_xxx('cut', input_entity, tool_entity,
                                 removeObject=removeObject, removeTool=removeTool,
                                 delete=delete)


    def boolean_fragments(self, input_entity, tool_entity,
                          removeObject=False, removeTool=False, delete=False):
        return self._boolean_xxx('fragment', input_entity, tool_entity,
                                 removeObject=removeObject, removeTool=removeTool,
                                 delete=delete)
     
    def boolean_union2d(self, input_entity, tool_entity,
                      removeObject=False, removeTool=False, delete=False):
       
        def get_dimtag(entity):
           dimtags = []
           for en in entity:
               dimtags.append(id2dimtag(en))
           return dimtags
       

        all_entity = input_entity + tool_entity     
        out_entity = self._boolean_xxx('fuse', input_entity, tool_entity,
                                 removeObject=removeObject, removeTool=removeTool,
                                 delete=delete)
        self.factory.synchronize()                

        xmax = -np.inf
        xmin =  np.inf
        ymax = -np.inf
        ymin =  np.inf
        zmax = -np.inf
        zmin =  np.inf
        
        def update_maxmin(dim, tag, xmin, ymin, zmin, xmax, ymax, zmax):
            x1, y1, z1, x2, y2, z2 = self.model.getBoundingBox(dim, tag)           
            xmax = np.max([xmax, x2])
            ymax = np.max([ymax, y2])
            zmax = np.max([zmax, z2])
            xmin = np.min([xmin, x1])
            ymin = np.min([ymin, y1])
            zmin = np.min([zmin, z1])
            return xmin, ymin, zmin, xmax, ymax, zmax
        
        out_dimtag = get_dimtag(out_entity)
        for dim, tag in out_dimtag:
            xmin, ymin, zmin, xmax, ymax, zmax = update_maxmin(dim, tag,
                                                               xmin, ymin, zmin,
                                                               xmax, ymax, zmax)
        dprint1("bounding box", xmin, ymin, zmin, xmax, ymax, zmax)
        
        dx = xmax-xmin
        dy = ymax-ymin
        bbx = self.factory.addRectangle(xmin-dx/10., ymin-dy/10., (zmin+zmax)/2.,
                                        dx*1.2, dy*1.2)
        out_dimtag2, dimtagMap = self.factory.cut(((2, bbx),), out_dimtag)
        #print(out_dimtag2)
        bbx = self.factory.addRectangle(xmin-dx/10., ymin-dy/10., (zmin+zmax)/2.,
                                        dx*1.2, dy*1.2)
        out_dimtag3, dimtagMap = self.factory.cut(((2,bbx),), out_dimtag2)
        self.factory.synchronize()                       
        return dimtag2id(out_dimtag3)                        

    def apply_fragments(self):
        self.factory.synchronize()        
        if self.dim == 0: return

        dimtags =  self.model.getEntities(self.dim)
        if len(dimtags) != 1:
            self.factory.fragment(dimtags[:1], dimtags[1:],
                                  removeObject=True, removeTool=True)
            
            self.factory.synchronize()
            return
        
        if self.dim > 1:
           dimtags =  self.model.getEntities(self.dim-1)
           if len(dimtags) != 1:
               self.factory.fragment(dimtags[:1], dimtags[1:],
                                  removeObject=True, removeTool=True)
               self.factory.synchronize()
               return
           
        if self.dim > 2:
           dimtags =  self.model.getEntities(self.dim-2)
           if len(dimtags) != 1:
               print('doing this 3')                              
               self.factory.fragment(dimtags[:1], dimtags[1:],
                                  removeObject=True, removeTool=True)
               self.factory.synchronize()
               
                             
    def remove(self, entity, recursive=False):
        dimtags = []
        for en in entity:
            dimtags.append(id2dimtag(en))
        self.factory.remove(dimtags, recursive=recursive)
        return []
     
    def copy(self, entity):
        dimtags = []
        for en in entity:
            dimtags.append(id2dimtag(en))
        dimtags2 = self.factory.copy(dimtags)
        return dimtag2id(dimtags2)                                
        
    def rotate(self, entity, x, y, z, ax, ay, az, angle):
        dimtags = []
        for en in entity:
            dimtags.append(id2dimtag(en))
        self.factory.rotate(dimtags, x, y, z, ax, ay, az, angle)
        return []

    def translate(self, entity, dx, dy, dz):
        dimtags = []
        for en in entity:
            dimtags.append(id2dimtag(en))
        self.factory.translate(dimtags, dx, dy, dz)
        return []

    def dilate(self, entity, x, y, z, a, b, c):
        dimtags = []
        for en in entity:
            dimtags.append(id2dimtag(en))
        self.factory.dilate(dimtags, x, y, z, a, b, c)
        return []
     
    def symmetrize(self, entity, a, b, c, d):
        dimtags = []
        for en in entity:
            dimtags.append(id2dimtag(en))
        self.factory.symmetrize(dimtags, a, b, c, d)
        return []

    def extrude(self, entity, translation_axis=None,
                point_on_axis=None,
                rotation_axis=None,
                angle = 0):
       
        #for en in entity:
        dimtags = [id2dimtag(entity)]
        if translation_axis is not None:
            tax = translation_axis
            dimtags2 = self.factory.extrude(dimtags, tax[0], tax[1], tax[2],)
        else:
            pax = point_on_axis
            rax = rotation_axis
            dimtags2 = self.factory.revolve(dimtags, pax[0], pax[1], pax[2],
                                          rax[0], rax[1], rax[2], angle)
        dprint1("extrude out", dimtags2)          
        return dimtag2id(dimtags2)

    def call_synchronize(self):
        self.factory.synchronize()        
        #self.p = len(self.model.getEntities(0))

    def recursive_getBdry(self, dimtag, bdr=None):
        if bdr is None: bdr = []
        bb = self.model.getBoundary((dimtag,), oriented=False,)
        bdr.extend(bb)

        for x in bb:
           if x[0] > 0:
               bdr = self.recursive_getBdry(x, bdr=bdr)
        return bdr
       
    def get_unique_entity(self, entity):
        entity = [x for x in entity if isinstance(x, GeomIDBase)
                  and not isinstance(x, LineLoopID)
                  and not isinstance(x, SurfaceLoopID)]
        dimtags = get_dimtag(entity)
        dimtags = [x for x in sorted(dimtags)]

        self.factory.synchronize()

        allent = self.model.getEntities()
        #print('all', allent)
        dimtags = [x for x in dimtags if x in allent]
        outdimtags = dimtags[:]

        #print('input', dimtags)
        for dimtag in reversed(dimtags):
           bdimtags = self.recursive_getBdry(dimtag)
           print("checking", dimtag, bdimtags)
           for x in bdimtags:
               if x in outdimtags:
                   idx = outdimtags.index(x)
                   del outdimtags[idx]
               if not x in allent:
                   idx = outdimtags.index(x)
                   del outdimtags[idx]
        #print('output', outdimtags)                                                     
        return dimtag2id(outdimtags)

    '''
    high level interface: 
      methods below directroy corresponds to GUI interface
    '''
    def add_sequence(self, gui_name, gui_param, geom_name):
        self.geom_sequence.append((gui_name, gui_param, geom_name))

    def Point_build_geom(self, objs, *args):
        xarr, yarr, zarr = args
        lcar = 0.0
        try:
           pos = np.vstack((xarr, yarr, zarr)).transpose()
        except:
           print("can not make proper input array")
           return
        PTs = [self.add_point(p, lcar=lcar) for p in pos]
        # apparently I should use this object (poly.surface)...?
        _newobjs = []        
        for p in PTs:
           newkey = objs.addobj(p, 'pt')
           _newobjs.append(newkey)
           
        return  objs.keys(), _newobjs

    def Line_build_geom(self, objs, *args):
        xarr, yarr, zarr, make_spline = args
        lcar = 0.0
        if len(xarr) < 2: return
        try:
           pos = np.vstack((xarr, yarr, zarr)).transpose()
        except:
           print("can not make proper input array")
           return

        dist = np.sqrt(np.sum((pos[:-1,:]- pos[1:,:])**2,1))
        if min(dist) == 0.0:
           assert False, "minimum distance between point is 0.0"
        if max(dist) > min(dist)*1e4:
           assert False, "some points are too close (d_max > d_min*1e4)"

        
        pt1 = self.add_point(pos[0], lcar, mask=True)
        pt2 = self.add_point(pos[-1], lcar, mask=True)        

        pts = [pt1]
        for ii, p in enumerate(pos[1:-1]):
            pt = self.add_point(p, lcar, mask = False)
            pts.append(pt)
        pts.append(pt2)

        if not make_spline:
            pts1 = pts[:-1]
            pts2 = pts[1:]

            newkeys = []
            for p1, p2 in zip(pts1, pts2):
                ln = self.add_line(p1, p2)
                newkeys.append(objs.addobj(ln, 'ln'))
            _newobjs = newkeys
        else:     
            spline = self.add_spline(pts, remove_control=True)
            newobj = objs.addobj(spline, 'sp')
            _newobjs = [newobj]
            
        newobj1 = objs.addobj(pts[0], 'pt')
        newobj2 = objs.addobj(pts[-1], 'pt')
        #print("entities(0)", geom.model.getEntities())                     
        
        _newobjs.append(newobj1)
        _newobjs.append(newobj2)

        return  objs.keys(), _newobjs

    def Polygon_build_geom(self, objs, *args):
        xarr, yarr, zarr = args
        lcar = 0.0
        if len(xarr) < 2: return
        try:
           pos = np.vstack((xarr, yarr, zarr)).transpose()
        except:
           print("can not make proper input array")
           return
        # check if data is already closed...
        if np.abs(np.sum((pos[0] - pos[-1])**2)) < 1e-17:
            pos = pos[:-1]
        poly = self.add_polygon(pos, lcar = lcar)

        # apparently I should use this object (poly.surface)...?
        newkey = objs.addobj(poly.surface, 'pol')
        return  objs.keys(), [newkey]

    def Spline_build_geom(self, objs, *args):
        pts = args
        pts = [x.strip() for x in pts[0].split(',')]
        
        #pts = [objs[x] for x in pts]
        pts = get_target1(objs, pts, 'p')        
        spline = self.add_spline(pts)
        newkey = objs.addobj(spline, 'sp')
        
        return  objs.keys(), [newkey]        

    def CreateLine_build_geom(self, objs, *args):
        pts = args
        pts = [x.strip() for x in pts[0].split(',')]        
        pts = get_target1(objs, pts, 'p')
        pts0 = pts[:-1]
        pts1 = pts[1:]

        newkeys = []
        
        for p0, p1 in zip(pts0, pts1):
             #if not p0 in objs:
             #    assert False, p0 + " does not exist"
             #if not p1 in objs:
             #    assert False, p1 + " does not exist"
             line = self.add_line(p0, p1)
             newkeys.append(objs.addobj(line, 'ln'))

        return  objs.keys(), newkeys

    def LineLoop_build_geom(self, objs, *args):
        pts = args
        pts = [x.strip() for x in pts[0].split(',')]

        ptx = get_target1(objs, pts, 'l')
        #pts = [(objs[x] if not x.startswith('-') else objs[x[1:]]) for x in pts]
        for x in pts:
           if x.startswith('-'):
               if x[1:] in objs:
                   del objs[x[1:]]
           else:
               if x in objs:
                   del objs[x]
               
        lloop = self.add_line_loop(ptx)
        newkey = objs.addobj(lloop, 'll')
        
        return  objs.keys(), [newkey]

    def CreateSurface_build_geom(self, objs, *args):
        pts, isFilling = args
        pts = [x.strip() for x in pts.split(',')]
        
        ptx = get_target1(objs, pts, 'l')
        #pts = [(objs[x] if not x.startswith('-') else objs[x[1:]]) for x in pts]
        for x in pts:
           if x.startswith('-'):
               if x[1:] in objs:
                   del objs[x[1:]]
           else:
               if x in objs:
                   del objs[x]
                   
        #objid = [(objs[x] if not x.startswith('-') else objs[x[1:]]) for x in pts]
        #objsign = [not x.startswith('-') for x in pts]        
        #for x in pts:
        #   if x.startswith('-'): del objs[x[1:]]
        #   else: del objs[x]
           
        if isFilling:
           surface = self.add_surface_filling(ptx)
           newobj1 = objs.addobj(surface, 'sf')
           newkeys = [newobj1]           
        else:
           ll = self.add_line_loop(ptx)
           #newobj1 = objs.addobj(ll, 'll')
           surface = self.add_plane_surface(ll)            
           newobj2 = objs.addobj(surface, 'ps')
           newkeys = [newobj2]

        return  objs.keys(), newkeys        

    def SurfaceLoop_build_geom(self, objs, *args):
        pts = args
        pts = [x.strip() for x in pts[0].split(',')]
        #pts = [(objs[x] if not x.startswith('-') else objs[x[1:]]) for x in pts]
        ptx = get_target(objs, targets, 'f')        
        sl = self.add_surface_loop(pts)
        newobj = objs.addobj(sl, 'sl')
        
        return  objs.keys(), [newobj]

    def CreateVolume_build_geom(self, objs, *args):
        pts = args
        pts = [x.strip() for x in pts[0].split(',')]

        ptx = get_target(objs, targets, 'f')                
        sl = self.add_surface_loop(pts)
        #newobj1 = objs.addobj(sl, 'sl')
        vol = self.add_volume(sl)
        newobj2 = objs.addobj(vol, 'vol')

        return  objs.keys(), [newobj2]   
    
    def Rect_build_geom(self, objs, *args):
        c1,  e1,  e2 = args
        lcar = 0.0
        
        c1 = np.array(c1);
        e1 = np.array(e1);
        e2 = np.array(e2);
        p1 = self.add_point(c1, lcar)
        p2 = self.add_point(c1+e1, lcar)
        p3 = self.add_point(c1+e1+e2, lcar)
        p4 = self.add_point(c1+e2, lcar)
        l1 = self.add_line(p1, p2)
        l2 = self.add_line(p2, p3)
        l3 = self.add_line(p3, p4)
        l4 = self.add_line(p4, p1)        
        ll1 = self.add_line_loop([l1, l2, l3, l4])
        rec1 = self.add_plane_surface(ll1)
        newkey = objs.addobj(rec1, 'rec')
        return  objs.keys(), [newkey]
    
        #self._objkeys = objs.keys()
        #self._newobjs = newkeys
        
    def Circle_build_geom(self, objs, *args):
        center, ax1, ax2, radius = args
        lcar = 0.0
        a1 = np.array(ax1);  a2 = np.array(ax2)
        a2 = np.cross(np.cross(a1, a2), a1)
        a1 = a1/np.sqrt(np.sum(a1**2))*radius
        a2 = a2/np.sqrt(np.sum(a2**2))*radius                      

        c =np.array(center)
        p1 = self.add_point(c+a1, lcar)
        p2 = self.add_point(c+a2, lcar)
        p3 = self.add_point(c-a1, lcar)
        p4 = self.add_point(c-a2, lcar)                      
        pc = self.add_point(c, lcar)
        ca1 = self.add_circle_arc(p1, pc, p2)
        ca2 = self.add_circle_arc(p2, pc, p3)
        ca3 = self.add_circle_arc(p3, pc, p4)
        ca4 = self.add_circle_arc(p4, pc, p1)
        ll1 = self.add_line_loop([ca1, ca2, ca3, ca4])
        ps1 = self.add_plane_surface(ll1)
        newkey = objs.addobj(ps1, 'ps')
        return  objs.keys(), [newkey]
    
        #self._objkeys = objs.keys()
        #self._newobjs = [newkey]        

    def Box_build_geom(self, objs, *args):
        c1,  e1,  e2,  e3 = args
        lcar = 0.0
        c1 = np.array(c1);
        e1 = np.array(e1);
        e2 = np.array(e2);
        p1 = self.add_point(c1, lcar)
        p2 = self.add_point(c1+e1, lcar)
        p3 = self.add_point(c1+e2, lcar)
        p4 = self.add_point(c1+e3, lcar)
        p5 = self.add_point(c1+e1+e2, lcar)        
        p6 = self.add_point(c1+e2+e3, lcar)
        p7 = self.add_point(c1+e3+e1, lcar)
        p8 = self.add_point(c1+e3+e2+e1, lcar)
        
        l1 = self.add_line(p1, p2)
        l2 = self.add_line(p2, p5)
        l3 = self.add_line(p5, p3)
        l4 = self.add_line(p3, p1)
        l5 = self.add_line(p1, p4)
        l6 = self.add_line(p2, p7)
        l7 = self.add_line(p5, p8)
        l8 = self.add_line(p3, p6)
        l9  = self.add_line(p4, p7)
        l10 = self.add_line(p7, p8)
        l11 = self.add_line(p8, p6)        
        l12 = self.add_line(p6, p4)
        
        
        ll1 = self.add_curve_loop([l1, l2, l3, l4])
        ll2 = self.add_curve_loop([l5, l9, l6, l1])        
        ll3 = self.add_curve_loop([l6, l10, l7, l2])
        ll4 = self.add_curve_loop([l7, l11, l8, l3])        
        ll5 = self.add_curve_loop([l8, l12, l5, l4])
        ll6 = self.add_curve_loop([l9, l10, l11, l12])
        
        rec1 = self.add_plane_surface(ll1)
        rec2 = self.add_plane_surface(ll2)
        rec3 = self.add_plane_surface(ll3)
        rec4 = self.add_plane_surface(ll4)
        rec5 = self.add_plane_surface(ll5)
        rec5 = self.add_plane_surface(ll6)

        sl = self.add_surface_loop([ll1, ll2, ll3, ll4, ll5, ll6])
        v1 = self.add_volume(sl)
        
        newkey = objs.addobj(v1, 'bx')
        
        return  objs.keys(), [newkey]
    
    def Ball_build_geom(self, objs, *args):
        self.factory.synchronize()

        x0,  l1,  l2,  l3 =  args
        lcar = 0.0
        radii = [l1, l2, l3]
        rr = min(radii)
        
        '''
        v1 = self.factory.addSphere(x0[0], x0[1], x0[2], rr, angle1=-np.pi, angle2 = 0)
        #v2 = self.factory.addSphere(x, y, z, radius, angle1= 0, angle2 = np.pi)                
        #v1 = self.add_sphere(x0[0], x0[1], x0[2], rr)
        
        if (l1/rr != 1.0 or l2/rr != 1.0 or l3/rr != 1.0):
            self.dilate([v1], x0[0], x0[1], x0[2], l1/rr, l2/rr, l3/rr)

        self.factory.synchronize()
        gmsh.write('hoge.brep')
        newkey = objs.addobj(v1, 'bl')
        return  objs.keys(), [newkey]                
        '''
        pc = self.add_point(x0, lcar)
        p1 = self.add_point([x0[0]+rr, x0[1], x0[2]], lcar=lcar)
        p2 = self.add_point([x0[0], x0[1]+rr, x0[2]], lcar=lcar)
        p3 = self.add_point([x0[0]-rr, x0[1], x0[2]], lcar=lcar)
        ca1 = self.add_circle_arc(p1, pc, p2)
        ca2 = self.add_circle_arc(p2, pc, p3)
        ln1 = self.add_line(p3, p1)
        ll1 = self.add_line_loop([ca1, ca2, ln1])
        ps1 = self.add_plane_surface(ll1)
        dst = [id2dimtag(ps1), ]

        volumes = []
        for i in range(4):
           ret = self.factory.revolve(dst,
                                   x0[0], x0[1], x0[2],
                                    1, 0, 0, np.pi/2.)
           dst = ret[:1]
           volumes.append(ret[1])

        ret = self.factory.fuse(volumes[:1], volumes[1:])
        v1 = VolumeID(ret[0][0][1])
        
        if (l1/rr != 1.0 or l2/rr != 1.0 or l3/rr != 1.0):
            self.dilate([v1], x0[0], x0[1], x0[2], l1/rr, l2/rr, l3/rr)
        
        newkey = objs.addobj(v1, 'bl')
        return  objs.keys(), [newkey]

    def Cone_build_geom(self, objs, *args):
        x0,  d0,  r1, r2, angle = args
        
        an = angle if angle < 180 else angle/2
            
        v1 = self.add_cone(x0[0], x0[1], x0[2], d0[0], d0[1], d0[2],
                           r1, r2,  an/180*np.pi)
        if angle >=180:
           v2 = self.add_cone(x0[0], x0[1], x0[2], d0[0], d0[1], d0[2],
                              r1, r2,  an/180*np.pi)
           v2 = [id2dimtag(v2), ]           
           self.factory.rotate(v2, x0[0], x0[1], x0[2],
                               d0[0], d0[1], d0[2],  an/180*np.pi)
           v1 = [id2dimtag(v1), ]                      
           ret = self.factory.fuse(v1, v2)
           v1 = VolumeID(ret[0][0][1])

        newkey = objs.addobj(v1, 'cn')
        return  objs.keys(), [newkey]

    def Cylinder_build_geom(self, objs, *args):
        x0,  d0,  r1,  angle = args
        lcar = 0.0
        d0 = np.array(d0)
        if np.sum(d0*np.array([1,0,0])) > np.sum(d0*np.array([0,1,0])):
           a1 = np.cross(d0, [0, 1, 0])
        else:
           a1 = np.cross(d0, [1, 0, 0])
        a2 = np.cross(d0, a1)   

        a1 = a1/np.sqrt(np.sum(a1**2))*r1
        a2 = a2/np.sqrt(np.sum(a2**2))*r1

        c =np.array(x0)
        p1 = self.add_point(c+a1, lcar)
        p2 = self.add_point(c+a2, lcar)
        p3 = self.add_point(c-a1, lcar)
        p4 = self.add_point(c-a2, lcar)                      
        pc = self.add_point(c, lcar)
        ca1 = self.add_circle_arc(p1, pc, p2)
        ca2 = self.add_circle_arc(p2, pc, p3)
        ca3 = self.add_circle_arc(p3, pc, p4)
        ca4 = self.add_circle_arc(p4, pc, p1)
        ll1 = self.add_line_loop([ca1, ca2, ca3, ca4])
        ps1 = self.add_plane_surface(ll1)
        
        ret = self.extrude(ps1, translation_axis=d0,)
        newkey = objs.addobj(ret[0], 'cn')
        return  objs.keys(), [newkey]

    def Wedge_build_geom(self, objs, *args):
        x0,  d0,  ltx = args
        v1 = self.add_wedge(x0[0], x0[1], x0[2], d0[0], d0[1], d0[2], ltx)
        
        newkey = objs.addobj(v1, 'wg')
        return  objs.keys(), [newkey]        

    def Torus_build_geom(self, objs, *args):
        x0,  r1,  r2, angle = args

        lcar = 0.0
        a1 = np.array([r2, 0, 0])
        a2 = np.array([0, 0, r2])        

        c =np.array(x0) + np.array([r1, 0, 0])
        p1 = self.add_point(c+a1, lcar)
        p2 = self.add_point(c+a2, lcar)
        p3 = self.add_point(c-a1, lcar)
        p4 = self.add_point(c-a2, lcar)                      
        pc = self.add_point(c, lcar)
        ca1 = self.add_circle_arc(p1, pc, p2)
        ca2 = self.add_circle_arc(p2, pc, p3)
        ca3 = self.add_circle_arc(p3, pc, p4)
        ca4 = self.add_circle_arc(p4, pc, p1)
        ll1 = self.add_line_loop([ca1, ca2, ca3, ca4])
        ps1 = self.add_plane_surface(ll1)

        dst = [id2dimtag(ps1), ]        
        volumes = []

        if angle > 270:
           seg = 4
        elif  angle > 180:
           seg = 3            
        elif  angle > 90:
           seg = 2                       
        else:
           seg = 1            
                
        an = angle/seg
        
        for i in range(seg):
            ret = self.factory.revolve(dst,
                                      x0[0], x0[1], x0[2],
                                      0, 0, 1, np.pi/2.)
            dst = ret[:1]
            volumes.append(ret[1])

        if seg > 1:
            ret = self.factory.fuse(volumes[:1], volumes[1:])
            v1 = VolumeID(ret[0][0][1])
        else:
            v1 = VolumeID(ret[1][1])

        newkey = objs.addobj(v1, 'trs')
        return  objs.keys(), [newkey]                
    
    def Extrude_build_geom(self, objs, *args):
        targets,  tax, len = args
        targets = [x.strip() for x in targets.split(',')]
        targetID = get_target2(objs, targets)
        tax = tax/np.sqrt(np.sum(np.array(tax)**2))*len          
        newkeys = []
        for t, idd in zip(targets, targetID):
             #if not t in objs:
             #    assert False, t + " does not exist"
             ret = self.extrude(idd,
                          translation_axis=tax,)
                          #rotation_axis=rax,
                          #point_on_axis=pax
             from petram.geom.gmsh_geom_model import use_gmsh_api
             if use_gmsh_api:
                 newkeys.append(objs.addobj(ret[1], t))
                 newkeys.append(objs.addobj(ret[0], 'ex'))             
             else:
                 newkeys.append(objs.addobj(ret[0], t))
                 newkeys.append(objs.addobj(ret[1], 'ex'))             

        return  objs.keys(), newkeys
    
    def Revolve_build_geom(self, objs, *args):
        
        targets, pax, rax, angle = args
        targets = [x.strip() for x in targets.split(',')]
        targetID = get_target2(objs, targets)

        newkeys = []
        for t, idd in zip(targets, targetID):        
             #if not t in objs:
             #    assert False, t + " does not exist"
             ret = self.extrude(idd,
                                rotation_axis=rax,
                                point_on_axis=pax,
                                angle = angle*np.pi/180.)
             
             from petram.geom.gmsh_geom_model import use_gmsh_api
             if use_gmsh_api:
                 newkeys.append(objs.addobj(ret[1], t))
                 newkeys.append(objs.addobj(ret[0], 'ex'))             
             else:
                 newkeys.append(objs.addobj(ret[0], t))
                 newkeys.append(objs.addobj(ret[1], 'ex'))
                 
        return  objs.keys(), newkeys

    def Sweep_build_geom(self, objs, *args):
        print("objs", objs)
        targets, lines = args
        targets = [x.strip() for x in targets.split(',')]
        targetID = get_target2(objs, targets)
        lines = [x.strip() for x in lines.split(',')]        
        lineID = get_target2(objs, lines)

        newkeys = []
        for t, idd in zip(targets, targetID):        
             #if not t in objs:
             #    assert False, t + " does not exist"
             #dimtags = [id2dimtag(wireID)]                                         
             wire = self.factory.addWire(lineID)
             dimtags = [id2dimtag(idd)]                                         
             ret = self.factory.addPipe(dimtags, wire)
             newkeys.append(objs.addobj(ret[0], 'ex'))
                 
        return  objs.keys(), newkeys
    

    def Move_build_geom(self, objs, *args):          
        targets, dx, dy, dz, keep  = args
        targets = [x.strip() for x in targets.split(',')]
          
        newkeys = []
        tt = get_target2(objs, targets)                                

        if keep:
           tt = self.copy(tt)          
        self.translate(tt, dx, dy, dz)
        if keep:
            for t in tt:
                newkeys.append(objs.addobj(t, 'mv'))
                
        return  objs.keys(), newkeys

    def Rotate_build_geom(self, objs, *args):          
        targets, cc, aa,  angle, keep  = args
        cx, cy, cz = cc
        ax, ay, az = aa
        targets = [x.strip() for x in targets.split(',')]
          
        newkeys = []
        tt = get_target2(objs, targets)                        

        if keep:
           tt = self.copy(tt)          
        self.rotate(tt, cx, cy, cz, ax, ay, az, np.pi*angle/180.)
        if keep:
            for t in tt:
                newkeys.append(objs.addobj(t, 'rot'))
        return  objs.keys(), newkeys                

    def Scale_build_geom(self, objs, *args):          
        targets,  cc, ss, keep  = args
        cx, cy, cz = cc
        sx, sy, sz = ss
        targets = [x.strip() for x in targets.split(',')]
          
        newkeys = []
        tt = get_target2(objs, targets)                

        if keep:
           tt = self.copy(tt)          
        self.dilate(tt, cx, cy, cz, sx, sy, sz)
        if keep:
            for t in tt:
                newkeys.append(objs.addobj(t, 'sc'))          

        return  objs.keys(), newkeys

    def Array_build_geom(self, objs, *args):          
        targets, count, displacement  = args
        dx, dy, dz = displacement
        targets = [x.strip() for x in targets.split(',')]
          
        newkeys = []
        tt = get_target2(objs, targets)        

        for i in range(count):
           tt = self.copy(tt)          
           self.translate(tt, dx, dy, dz)
           for t in tt:
                newkeys.append(objs.addobj(t, 'cp'))
                
        return  objs.keys(), newkeys                

    def ArrayRot_build_geom(self, objs, *args):          
        targets, count, cc, aa,  angle = args
        cx, cy, cz = cc
        ax, ay, az = aa
        targets = [x.strip() for x in targets.split(',')]
          
        newkeys = []
        tt = get_target2(objs, targets)                        
        
        for i in range(count):
           tt = self.copy(tt)
           self.rotate(tt, cx, cy, cz, ax, ay, az, np.pi*angle/180.)           
           for t in tt:
                newkeys.append(objs.addobj(t, 'cp'))

        return  objs.keys(), newkeys                                 

    def Flip_build_geom(self, objs, *args):          
        targets, a, b, c, d,  keep  = args
        targets = [x.strip() for x in targets.split(',')]
          
        newkeys = []
        tt = get_target2(objs, targets)

        if keep:
           tt = self.copy(tt)          
        self.symmetrize(tt, a, b, c, d)
        if keep:
            for t in tt:
                newkeys.append(objs.addobj(t, 'flp'))

        return  objs.keys(), newkeys

    def Fillet_build_geom(self, objs, *args):
        volumes, curves, radius = args
        volumes = [x.strip() for x in volumes.split(',')]
        curves = [x.strip() for x in curves.split(',')]
        
        radii = [radius]

        volumes = get_target1(objs, volumes, 'v')
        curves  = get_target1(objs, curves, 'v')
        
        ret = self.fillet(volumes, curves, radii, removeVolume=True)
        newkeys = []
        for r in ret:
            newkeys.append(objs.addobj(r, 'vol'))

        return  objs.keys(), newkeys            

    def Chamfer_build_geom(self, objs, *args):
        volumes, curves, distances, surfaces  = args
        
        volumes = [x.strip() for x in volumes.split(',')]
        curves = [x.strip() for x in curves.split(',')]
        surfaces = [x.strip() for x in surfaces.split(',')]
        
        volumes = [objs[t] if t in objs else int(t)  for t in volumes]
        curves  = [objs[t] if t in objs else int(t)  for t in curves]
        surfaces = [objs[t] if t in objs else int(t)  for t in surfaces]
        ret = self.chamfer(volumes, curves, surfaces, distances, removeVolume=True)
        newkeys = []
        for r in ret:
            newkeys.append(objs.addobj(r, 'vol'))
        
        return  objs.keys(), newkeys

    def Copy_build_geom(self, objs, *args):
        targets  = args[0]
        targets = [x.strip() for x in targets.split(',')]

        newkeys = []
        tt = get_target2(objs, targets)
        ret = self.copy(tt)
        for r in ret:
            newkeys.append(objs.addobj(r, 'cp'))

        return  objs.keys(), newkeys

    def Remove_build_geom(self, objs, *args):
        targets, recursive = args
        targets = [x.strip() for x in targets.split(',')]

        newkeys = []
        tt = get_target2(objs, targets)
        self.remove(tt, recursive=recursive)
        for t in targets:
           if t in objs: del objs[t]

        return  objs.keys(), newkeys           
    
    def Difference_build_geom(self, objs, *args):
        tp, tm, delete_input, delete_tool = args
        tp = [x.strip() for x in tp.split(',')]
        tm = [x.strip() for x in tm.split(',')]          

        input_entity = get_target2(objs, tp)
        tool_entity  = get_target2(objs, tm)
        
        ret = self.boolean_difference(
                          input_entity,
                          tool_entity,
                          removeObject = delete_input,
                          removeTool = delete_tool)

        newkeys = []
        newkeys.append(objs.addobj(ret[0], 'diff'))
        if len(ret) > 1:
            for o in ret[1:]:
                newkeys.append(objs.addobj(o,  get_geom_key(o)))
                
        if delete_input:
            for x in tp[:1]: 
                if x in objs: del objs[x]          
        if delete_tool:
            for x in tp[1:]: 
                if x in objs: del objs[x]          
            
        return  objs.keys(), newkeys

    def Union_build_geom(self, objs, *args):
        tp, delete_input, delete_tool = args
        tp = [x.strip() for x in tp.split(',')]
        if len(tp) < 2: return

        input_entity = get_target2(objs, tp[:1])
        tool_entity  = get_target2(objs, tp[1:])
        ret = self.boolean_union(
                          input_entity,
                          tool_entity,
                          removeObject = delete_input,
                          removeTool = delete_tool)
        
        newkeys = []
        newkeys.append(objs.addobj(ret[0], 'uni'))
        if len(ret) > 1:
            for o in ret[1:]:
                newkeys.append(objs.addobj(o,  get_geom_key(o)))
                
        if delete_input:
           for x in tp[:1]:
             if x in objs: del objs[x]
        if delete_tool:
           for x in tp[1:]: 
             if x in objs: del objs[x]
             
        return  objs.keys(), newkeys             

    def Union2D_build_geom(self, objs, *args):
        tp, delete_input, delete_tool = args        
        tp = [x.strip() for x in tp.split(',')]
        if len(tp) < 2: return

        input_entity = get_target2(objs, tp[:1])
        tool_entity  = get_target2(objs, tp[1:])
        ret = self.boolean_union2d(
                          input_entity,
                          tool_entity,
                          removeObject = delete_input,
                          removeTool = delete_tool)
        
        newkeys = []
        newkeys.append(objs.addobj(ret[0], 'uni'))
        if len(ret) > 1:
            for o in ret[1:]:
                newkeys.append(objs.addobj(o,  get_geom_key(o)))
                
        if delete_input:
            for x in tp[:1]: 
                if x in objs: del objs[x]          

        if delete_tool:
            for x in tp[1:]: 
                if x in objs: del objs[x]          

        return  objs.keys(), newkeys                             

    def Intersection_build_geom(self, objs, *args):
        tp, delete_input, delete_tool = args        
        tp = [x.strip() for x in tp.split(',')]
        if len(tp) < 2: return

        input_entity = get_target2(objs, tp[:1])
        tool_entity  = get_target2(objs, tp[1:])

        ret = self.boolean_intersection(
                          input_entity,
                          tool_entity,
                          removeObject = delete_input,
                          removeTool = delete_tool)
        newkeys = []
        newkeys.append(objs.addobj(ret[0], 'its'))
        if len(ret) > 1:
            for o in ret[1:]:
                newkeys.append(objs.addobj(o,  get_geom_key(o)))

        if delete_input:
            for x in tp[:1]: 
                if x in objs: del objs[x]          
        if delete_tool:
            for x in tp[1:]: 
                if x in objs: del objs[x]          

        return  objs.keys(), newkeys
    
    def Fragments_build_geom(self, objs, *args):
        tp, delete_input, delete_tool = args
        tp = [x.strip() for x in tp.split(',')]
        if len(tp) < 2: return

        input_entity = get_target2(objs, tp[:1])
        tool_entity  = get_target2(objs, tp[1:])
        ret = self.boolean_fragments(
                          input_entity,
                          tool_entity,
                          removeObject = delete_input,
                          removeTool = delete_tool)

        newkeys = []
        newkeys.append(objs.addobj(ret[0], 'frag'))
        if len(ret) > 1:
            for o in ret[1:]:
                newkeys.append(objs.addobj(o, get_geom_key(o)))
                
        if delete_input:
            for x in tp[:1]: 
                if x in objs: del objs[x]          
        if delete_tool:
            for x in tp[1:]: 
                if x in objs: del objs[x]          
            
        return  objs.keys(), newkeys

    '''
    2D elements
    '''
    def Point2D_build_geom(self, objs, *args):
        xarr, yarr = args
        lcar = 0.0
        xarr = np.atleast_1d(xarr)
        yarr = np.atleast_1d(yarr)
        zarr = xarr * 0.0
        try:
           pos = np.vstack((xarr, yarr, zarr)).transpose()
        except:
           print("can not make proper input array")
           return
        PTs = [self.add_point(p, lcar=lcar) for p in pos]
        # apparently I should use this object (poly.surface)...?
        _newobjs = []        
        for p in PTs:
           newkey = objs.addobj(p, 'pt')
           _newobjs.append(newkey)
           
        return  objs.keys(), _newobjs

    # Define 2D version the same as 3D
    Line2D_build_geom = Line_build_geom
        
    def Circle2D_build_geom(self, objs, *args):
        center, ax1, ax2, radius = args
        lcar = 0.0
        a1 = np.array(ax1+[0])
        a2 = np.array(ax2+[0])
        a2 = np.cross(np.cross(a1, a2), a1)
        a1 = a1/np.sqrt(np.sum(a1**2))*radius
        a2 = a2/np.sqrt(np.sum(a2**2))*radius                      

        c =np.array(center+[0])
        p1 = self.add_point(c+a1, lcar)
        p2 = self.add_point(c+a2, lcar)
        p3 = self.add_point(c-a1, lcar)
        p4 = self.add_point(c-a2, lcar)                      
        pc = self.add_point(c, lcar)
        ca1 = self.add_circle_arc(p1, pc, p2)
        ca2 = self.add_circle_arc(p2, pc, p3)
        ca3 = self.add_circle_arc(p3, pc, p4)
        ca4 = self.add_circle_arc(p4, pc, p1)
        ll1 = self.add_line_loop([ca1, ca2, ca3, ca4])
        ps1 = self.add_plane_surface(ll1)
        newkey = objs.addobj(ps1, 'ps')

        return  objs.keys(), [newkey]

    def Arc2D_build_geom(self, objs, *args):
        center, ax1, ax2, radius, an1, an2, do_fill = args
        lcar = 0.0
        a1 = np.array(ax1+[0]);
        a2 = np.array(ax2+[0])
        a2 = np.cross(np.cross(a1, a2), a1)
        a1 = a1/np.sqrt(np.sum(a1**2))*radius
        a2 = a2/np.sqrt(np.sum(a2**2))*radius
        if an1 > an2:
           tmp = an2; an2 = an1; an1 = tmp
        if an2 - an1 > 180:
           assert False, "angle must be less than 180"

        an3 = (an1 + an2)/2.0
        pt1 = a1*np.cos(an1*np.pi/180.) + a2*np.sin(an1*np.pi/180.)
        pt2 = a1*np.cos(an2*np.pi/180.) + a2*np.sin(an2*np.pi/180.)
        pt3 = a1*np.cos(an3*np.pi/180.) + a2*np.sin(an3*np.pi/180.)        

        c =np.array(center+[0])
        p1 = self.add_point(c+pt1, lcar)
        p2 = self.add_point(c+pt2, lcar)
        p3 = self.add_point(c+pt3, lcar)        
        pc = self.add_point(c, lcar)
        ca1 = self.add_circle_arc(p1, pc, p3)
        ca2 = self.add_circle_arc(p3, pc, p2)

        if not do_fill:
            newkey1 = objs.addobj(ca1, 'ln')
            newkey2 = objs.addobj(ca2, 'ln')
            newkeys = [newkey1, newkey2]
        else:
            l1 = self.add_line(pc, p1)
            l2 = self.add_line(p2, pc)            
            ll1 = self.add_line_loop([l1, ca1, ca2, l2])
            ps1 =  self.add_plane_surface(ll1)
            newkeys = [objs.addobj(ps1, 'ps')]
            
        return  objs.keys(), newkeys

    def Rect2D_build_geom(self, objs, *args):
        c1,  e1,  e2 = args
        lcar = 0.0
        c1 = np.array(c1+[0]);
        e1 = np.array(e1+[0]);
        e2 = np.array(e2+[0]);
        p1 = self.add_point(c1, lcar)
        p2 = self.add_point(c1+e1, lcar)
        p3 = self.add_point(c1+e1+e2, lcar)
        p4 = self.add_point(c1+e2, lcar)
        l1 = self.add_line(p1, p2)
        l2 = self.add_line(p2, p3)
        l3 = self.add_line(p3, p4)
        l4 = self.add_line(p4, p1)        
        ll1 = self.add_line_loop([l1, l2, l3, l4])
        rec1 = self.add_plane_surface(ll1)
        newkey = objs.addobj(rec1, 'rec')
        return  objs.keys(), [newkey]


    def Polygon2D_build_geom(self, objs, *args):
        xarr, yarr = args
        zarr = [0]*len(yarr)
        lcar = 0.0
        if len(xarr) < 2: return
        try:
           pos = np.vstack((xarr, yarr, zarr)).transpose()
        except:
           print("can not make proper input array")
           return
        # check if data is already closed...
        if np.abs(np.sum((pos[0] - pos[-1])**2)) < 1e-17:
            pos = pos[:-1]
        poly = self.add_polygon(pos, lcar = lcar)

        # apparently I should use this object (poly.surface)...?
        newkey = objs.addobj(poly.surface, 'pol')

        return  objs.keys(), [newkey]        

    def Move2D_build_geom(self, objs, *args):          
        targets, dx, dy, keep  = args
        dz = 0.0
        targets = [x.strip() for x in targets.split(',')]
          
        newkeys = []
        tt = get_target2(objs, targets)                                
        if keep:
           tt = self.copy(tt)          
        self.translate(tt, dx, dy, dz)
        if keep:
            for t in tt:
                newkeys.append(objs.addobj(t, 'mv'))

        return  objs.keys(), newkeys                                
    
    def Rotate2D_build_geom(self, objs, *args):          
        targets, cc, angle, keep  = args
        cx, cy= cc; cz = 0.0
        ax, ay, az = 0.0, 0.0, 1.0
        targets = [x.strip() for x in targets.split(',')]
          
        newkeys = []
        tt = get_target2(objs, targets)                                        

        if keep:
           tt = self.copy(tt)          
        self.rotate(tt, cx, cy, cz, ax, ay, az, np.pi*angle/180.)
        if keep:
            for t in tt:
                newkeys.append(objs.addobj(t, 'rot'))
                
        return  objs.keys(), newkeys                
    
    def Flip2D_build_geom(self, objs, *args):          
        targets, a, b, d,  keep  = args
        c = 0.0
        targets = [x.strip() for x in targets.split(',')]
          
        newkeys = []
        tt = get_target2(objs, targets)        
        if keep:
           tt = self.copy(tt)          
        self.symmetrize(tt, a, b, c, d)
        if keep:
            for t in tt:
                newkeys.append(objs.addobj(t, 'flp'))          

        return  objs.keys(), newkeys
    
    def Scale2D_build_geom(self, objs, *args):          
        targets,  cc, ss, keep  = args
        cx, cy = cc; cz = 0.0
        sx, sy = ss; sz = 1.0
        targets = [x.strip() for x in targets.split(',')]
          
        newkeys = []
        tt = get_target2(objs, targets)                
        if keep:
           tt = self.copy(tt)          
        self.dilate(tt, cx, cy, cz, sx, sy, sz)
        if keep:
            for t in tt:
                newkeys.append(objs.addobj(t, 'sc'))          

        return  objs.keys(), newkeys
    
    def Array2D_build_geom(self, objs, *args):          
        targets, count, displacement  = args
        dx, dy = displacement
        dz = 0.0
        targets = [x.strip() for x in targets.split(',')]
          
        newkeys = []
        tt = get_target2(objs, targets)                        
        for i in range(count):
           tt = self.copy(tt)          
           self.translate(tt, dx, dy, dz)
           for t in tt:
                newkeys.append(objs.addobj(t, 'cp'))
                
        return  objs.keys(), newkeys
    
    def WorkPlane_build_geom(self, objs, *args):
        c1,  a1,  a2 = args
        c1 = np.array(c1)
        a1 = np.array(a1); a1 = a1/np.sqrt(np.sum(a1**2))
        a2 = np.array(a2); a2 = a2/np.sqrt(np.sum(a2**2))        


        ax = np.cross(np.array([1., 0., 0.]), a1)
        an = np.arcsin(np.sqrt(np.sum(ax**2)))
        
        tt = [objs[t] for t in objs]

        from petram.geom.gmsh_geom_wrapper import VertexID, LineID, SurfaceID

        #print("tt_in", tt)
        tt = self.get_unique_entity(tt)
        #dprint1("tt_out", tt)        
        #print("entities(1)", geom.model.getEntities())
        #for t in tt:
        #     if isinstance(t, SurfaceID): continue
        #     print("working on t", t)             
        #     geom.translate([t], c1[0], c1[1], c1[2])
        #     geom.factory.synchronize()
        #     print("entities(-1)", geom.model.getEntities())             
        #     geom.rotate([t], 0, 0, 0, ax[0], ax[1], ax[2], an)
        self.translate(tt, c1[0], c1[1], c1[2])
        if np.sum(ax**2) != 0.0 and an != 0.0:
            #print(ax, an)
            self.rotate(tt, 0, 0, 0, ax[0], ax[1], ax[2], an)
            
        from petram.geom.geom_utils import rotation_mat
        R = rotation_mat(ax, an)
        '''
        c = np.cos(an); s = np.sin(an)
        R = np.array(
            [[c + (1-c)*ax[0]**2, ax[0]*ax[1]*(1-c)-ax[2]*s, ax[0]*ax[2]*(1-c)+ax[1]*s],
             [ax[0]*ax[1]*(1-c)+ax[2]*s, c + (1-c)*ax[1]**2,  ax[1]*ax[2]*(1-c)-ax[0]*s],
             [ax[0]*ax[2]*(1-c)-ax[1]*s, ax[1]*ax[2]*(1-c)+ax[0]*s, c + (1-c)*ax[2]**2]]
            )
        '''
        y2 =  np.dot(R, np.array([0, 1, 0]))
                     
        ax = np.cross(y2, a2)
        an = np.arcsin(np.sqrt(np.sum(ax**2)))
        
        #for t in tt:
        #     if isinstance(t, SurfaceID): continue
        #     print("working on t", t)             
        #
        #     geom.rotate([t], 0, 0, 0, ax[0], ax[1], ax[2], an)
        if np.sum(ax**2) != 0.0 and an != 0.0:
            #print(ax, an)          
            self.rotate(tt, 0, 0, 0, ax[0], ax[1], ax[2], an)

        #self._newobjs = objs.keys()
        return  objs.keys(), []

    def BrepImport_build_geom(self, objs, *args):
        cad_file = args[0]
        highestDimOnly = True
        
        PTs = self.factory.importShapes(cad_file, 
                                        highestDimOnly=highestDimOnly)
        
        # apparently I should use this object (poly.surface)...?

        newkeys = []

        dim = max([p[0] for p in PTs])
        
        for p in PTs:
            if p[0] == dim:
               pp = dimtag2id([p])
               newkeys.append(objs.addobj(pp[0], 'impt'))

        return  objs.keys(), newkeys
    
    def CADImport_build_geom(self, objs, *args):
        return self.BrepImport_build_geom(objs, *args)
        
    '''
    sequence/preview/brep generator
    '''
    def run_sequence(self):
        from petram.geom.gmsh_geom_model import GeomObjs
        objs = GeomObjs()
        gui_data = dict()

        isWP = False
        
        for gui_name, gui_param, geom_name in self.geom_sequence:
            if self.logfile is not None:
                self.logfile.write("processing " + gui_name + "\n")
                self.logfile.write("data " + str(geom_name) + ":" + str(gui_param) + "\n")
            if self.queue is not None:
                self.queue.put((False, "processing " + gui_name))
            
            if geom_name == "WP_Start":
                tmp = objs.duplicate()
                org_keys = objs.keys()

                for x in org_keys: del tmp[x]
                
                org_objs = objs
                objs = tmp
                isWP = True
                
            elif geom_name == "WP_End":
                for x in objs: org_objs[x] = objs[x]
                objs = org_objs
                isWP = False                

            else:
                try:
                    method = getattr(self, geom_name+'_build_geom')
                    objkeys, newobjs = method(objs, *gui_param)
                    gui_data[gui_name] = (objkeys, newobjs)
                except:
                    import traceback
                    if self.logfile is not None:                    
                        self.logfile.write("failed " + traceback.format_exc())
                    assert False, traceback.format_exc()

        #cacheName = "" if isWP else gui_name
        return gui_data, objs
    

    def generate_preview_mesh(self, filename = ''):
        
        if self.queue is not None:
            self.queue.put((False, "generating preview"))

        ss = self.getObjSizes()
        
        dim2_size = min([s[2] for s in ss if s[0]==2]+[3e20])
        dim1_size = min([s[2] for s in ss if s[0]==1]+[3e20])

        xmin, xmax, ymin, ymax, zmin,zmax = self.getBoundingBox()
        modelsize = ((xmax-xmin)**2 + (ymax-ymin)**2 + (zmax-zmin)**2)**0.5

        vcl = self.getVertexCL()
        for tag in vcl:
           gmsh.model.mesh.setSize(((0, tag),), vcl[tag]/2.5)


        #print(geom.model.getEntities())
        #geom.model.setVisibility(((3,7), ), False, True)

        #gmsh.option.setNumber("Mesh.CharacteristicLengthMax", dim2_size/3.)
        #gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)
        #gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 1)
        #gmsh.option.setNumber("Mesh.MeshOnlyVisible", 1)
        #gmsh.option.setNumber("Mesh.Mesh.CharacteristicLengthFromCurvature", 1)

        gmsh.option.setNumber("Mesh.MaxNumThreads1D", self.maxthreads)
        gmsh.option.setNumber("Mesh.MaxNumThreads2D", self.maxthreads)

        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", modelsize/self.geom_prev_res)
        gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 1)                
        gmsh.model.mesh.generate(1)
        
        gmsh.option.setNumber("Mesh.Algorithm", self.geom_prev_algorithm)
        #gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 1e22)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", modelsize/10)        
        gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)        
        gmsh.model.mesh.generate(2)
        
        if filename != '':
            import os
            geom_msh = os.path.join(os.getcwd(), filename+'.msh')
            gmsh.write(geom_msh)
    
    def generate_brep(self, objs, filename = '', finalize=False):
        
        if finalize:
            if self.logfile is not None:
                self.logfile.write("finalize is on : computing  fragments\n")
            if self.queue is not None:
                self.queue.put((False, "finalize is on : computing  fragments"))
                
            self.apply_fragments()
            
        self.factory.synchronize()

        '''
        Save BREP for meshing.
        '''
        import os

        map = self.getEntityNumberingInfo()
        
        geom_brep = os.path.join(os.getcwd(), filename+'.brep')
        gmsh.write(geom_brep)

        do_map_always = True
        if finalize or do_map_always:
            '''
            We need to reload it here so that indexing is consistent
            in meshing.
            '''
            gmsh.clear()
            gmsh.model.occ.importShapes(geom_brep, highestDimOnly=False)
            gmsh.model.occ.synchronize()

            self.applyEntityNumberingInfo(map, objs)
            
        return geom_brep
        
    def run_generator(self, no_mesh=False, finalize=False, filename = '',
                      progressbar = None):
        
        kwargs = {'PreviewResolutio': self.geom_prev_res,
                  'PreviewAlgorithm': self.geom_prev_algorithm,
                  'OCCParallel': self.occ_parallel,
                  'Maxthreads': self.maxthreads}
                 
        q = mp.Queue()
        p = mp.Process(target = generator,
                       args = (q, self.geom_sequence, no_mesh,
                               finalize, filename, kwargs))
        p.start()
        logfile = q.get(True)
        dprint1("log file: ", logfile)

        istep = 0
        
        while True:
            try:
                ret = q.get(True, 1)
                if ret[0]: break
                else:
                    dprint1(ret[1])
                    
                if progressbar is not None:
                    istep += 1
                    progressbar.Update(istep, newmsg=ret[1])                    
                
            except QueueEmpty:
                if not p.is_alive():
                    if progressbar is not None:                    
                       progressbar.Destroy()
                    assert False, "Child Process Died"
                    break
                time.sleep(1.)                    
                if progressbar is not None:
                    import wx
                    wx.Yield()
                    if progressbar.WasCancelled():
                       if p.is_alive():
                           p.terminate()
                       progressbar.Destroy()
                       assert False, "Geometry Generation Aborted"
                    
            time.sleep(0.03)
        return ret[1]
        

def generator(q, sequence, no_mesh, finalize, filename,  kwargs):
 
    
    kwargs['write_log'] = True
    kwargs['queue'] = q
    
    mw = Geometry(**kwargs)
    
    logfile = mw.logfile
    q.put((logfile.name))

    mw.geom_sequence = sequence
    mw.out_queue = q
    
    gui_data, objs= mw.run_sequence()

    if finalize:
        filename = filename
        brep_file = mw.generate_brep(objs, filename = filename, finalize = True)
    else:
        filename = sequrence[-1][0]
        mw.generate_brep(objs, filename = filename, finalize = False)
        brep_file = ''

    if no_mesh:
        q.put((True, (gui_data, objs, brep_file, None)))

    else:
        mw.generate_preview_mesh()

        from petram.geom.read_gmsh import read_pts_groups, read_loops        
        ptx, cells, cell_data = read_pts_groups(gmsh)
        l, s, v = read_loops(gmsh)
    
        data = ptx, cells, cell_data, l, s, v
        q.put((True, (gui_data, objs, brep_file, data)))


    
'''
   Not yet implemented...

   def addEllipse(x, y, z, r1, r2, tag=-1, angle1=0., angle2=2*pi):
   def addEllipseArc(startTag, centerTag, majorTag, endTag, tag=-1, nx=0., ny=0., nz=0.)
   def addBezier(pointTags, tag=-1)
   def addCurveLoop(curveTags, tag=-1)
   def addSurfaceFilling(wireTags, tag=-1, sphereCenterTag=-1)
   def twist(dimTags, x, y, z, dx, dy, dz, ax, ay, az, angle, numElements=[], heights=[], recombine=False)
   def addDisk(xc, yc, zc, rx, ry, tag=-1)
   def addSphere(xc, yc, zc, radius, tag=-1, angle1=-pi/2, angle2=pi/2, angle3=2*pi)
   def addBox(x, y, z, dx, dy, dz, tag=-1)

   def addThruSections(wireTags, tag=-1, makeSolid=True, makeRuled=False)
   def addThickSolid(volumeTag, excludeSurfaceTags, offset, tag=-1)
'''
