import numpy as np

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('GmshMeshActions')

from petram.phys.vtable import VtableElement, Vtable
from petram.mesh.gmsh_mesh_model import GmshMeshActionBase


def show(lines, gid, mode = "Line"):    
    if gid == "*":
        lines.append('Show "*";')
    elif gid == "": return
    else:
        gid = [str(x) for x in gid.split(',')]
        txt = 'Recursive Show {{ {}; }}'.format(
                mode + '{{{}}}'.format(','.join(gid)))
        lines.append(txt)

def hide(lines, gid, mode = "Line"):
    if gid == "*":
        lines.append('Hide "*";')
    elif gid == "": return        
    else:
        gid = [str(x) for x in gid.split(',')]
        if len(gid) == 0: return
        txt = 'Recursive Hide {{ {}; }}'.format(
                mode + '{{{}}}'.format(','.join(gid)))
        lines.append(txt)
        
def mesh(lines, dim = 1):
    lines.append('Mesh.MeshOnlyVisible=1;')
    lines.append('Mesh ' + str(dim) + ';')

def transfinite(lines, gid, mode = 'Line', nseg='',
                progression = 0, bump = 0):
    c = 'Transfinite '+mode
    if gid == "*":
        c += ' "*" = '
        print(c)
    else:
        gid = [str(x) for x in gid.split(',')]
        c += '{{ {} }}'.format(','.join(gid)) + ' = '
    c += str(nseg)
    if bump != 0:
        c += " Using Bump " + str(bump)
    if progression != 0:
        c += " Using Progression " + str(bump)
    
    lines.append(c+';')

class MeshData(object):
    def __init__(self, lines, num_entities):
        self.lines = lines
        
        self.done = {"Point":[],
                     "Line": [],
                     "Surface": [],
                     "Volume": []}
        self.num_entities = {"Point": num_entities[0],
                             "Line": num_entities[1],
                             "Surface": num_entities[2],
                             "Volume": num_entities[3],}
        
        
    def append(self, c):
        self.lines.append(c)

    def show_hide_gid(self, gid, mode = "Line"):
        if gid.strip() == "":
            return "" # dont do anything
        elif gid == "*":
            show(self, gid, mode = mode)            
        elif gid == "remaining":
            if self.done[mode] == "*": return "" # already all done
            show(self, "*", mode = mode)
            hide(self, ','.join([str(x) for x in self.done[mode]]),
                 mode = mode)            
        else:
            hide(self, "*", mode = mode)
            show(self, gid, mode = mode)
            
        if gid == "*":        
            self.done[mode] = "*"
        elif gid == "remaining":
            gid = self.get_remaining_txt(mode)
            self.done[mode] = "*"            
        else:
            gidnum = [int(x) for x in gid.split(',')]
            for x in gidnum:
                if not x in self.done[mode]:
                    self.done[mode].append(x)
        return gid
                    
    def get_remaining_txt(self, mode = "Line"):
        if self.done[mode] == "*": return ''
        ll = [x+1 for x in range(self.num_entities[mode])]
        for x in self.done[mode]:ll.remove(x)
        if len(ll) == 0:
            self.done[mode] = "*"
            return ''
        else:
            return ','.join([str(x) for x in ll])
            
        

data = (('geom_id', VtableElement('geom_id', type='string',
                                   guilabel = 'Line#',
                                   default = "remaining", 
                                   tip = "Line ID" )),
        ('num_seg', VtableElement('radius', type='float',
                                   guilabel = 'Number of segments',
                                   default = 5, 
                                   tip = "Number of segments" )),
        ('progression', VtableElement('progression', type='float',
                                   guilabel = 'Progression',
                                   default = 0, 
                                   tip = "Progression" )),
        ('bump', VtableElement('bump', type='float',
                                   guilabel = 'Bump',
                                   default = 0, 
                                   tip = "Bump" )),)

        
    
class TransfiniteLine(GmshMeshActionBase):
    vt = Vtable(data)    
    def build_mesh(self, lines):
        gid, nseg, p, b = self.vt.make_value_or_expression(self)
        gid = lines.show_hide_gid(gid, mode="Line")
        if gid == "": return
        transfinite(lines, gid, mode = 'Line', nseg=nseg,
                    progression = p,  bump = b)
        mesh(lines, dim = 1)            

