import numpy as np

gmsh_element_type = {
        15: 'vertex',
        1: 'line',
        2: 'triangle',
        3: 'quad',
        4: 'tetra',
        5: 'hexahedron',
        6: 'wedge',
        7: 'pyramid',
        8: 'line3',
        9: 'triangle6',
        10: 'quad9',
        11: 'tetra10',
        12: 'hexahedron27',
        13: 'prism18',
        14: 'pyramid14',
        26: 'line4',
        36: 'quad16',
        }
num_nodes_per_cell = {
    'vertex': 1,
    'line': 2,
    'triangle': 3,
    'quad': 4,
    'tetra': 4,
    'hexahedron': 8,
    'wedge': 6,
    'pyramid': 5,
    #
    'line3': 3,
    'triangle6': 6,
    'quad9': 9,
    'tetra10': 10,
    'hexahedron27': 27,
    'prism18': 18,
    'pyramid14': 14,
    'line4': 4,
    'quad16': 16,
    }

#dimtags =  gmsh.model.getEntities()
def read_loops(geom):
    model = geom.model
    
    model.occ.synchronize()
    v = {}
    s = {}
    l = {}
    
    dimtags =  model.getEntities(3)
    for dim, tag in dimtags:
        v[tag] = [y for x, y in model.getBoundary([(dim, tag)],
                                                       oriented=False)]
    dimtags =  model.getEntities(2)
    for dim, tag in dimtags:
        s[tag] = [y for x, y in model.getBoundary([(dim, tag)],
                                                       oriented=False)]
    dimtags =  model.getEntities(1)
    for dim, tag in dimtags:
        l[tag] = [y for x, y in model.getBoundary([(dim, tag)],
                                                       oriented=False)]
    return l, s, v

def read_loops2(geom):
    '''
    read vertex coordinats and loops together. 
    before calling this, do 
        self.hide_all()        
        gmsh.model.mesh.generate(1)
    '''
    l, s, v = read_loops(geom)
    model = geom.model
    nidx, coord, pcoord = geom.model.mesh.getNodes(dim=0)
    tags = [tag for dim, tag in geom.model.getEntities(0)]
    p = {t: nidx[k]-1  for k, t in enumerate(tags)}
    ptx = np.array(coord).reshape(-1, 3)
    return ptx, p, l, s, v

def read_pts_groups(geom, finished_lines=None, 
                          finished_faces=None):

    model = geom.model
    
    node_id, node_coords, parametric_coods =  model.mesh.getNodes()
    if len(node_coords) == 0:
        return np.array([]).reshape((-1,3)), {}, {}
    points = np.array(node_coords).reshape(-1, 3)

    node2idx = np.zeros(max(node_id)+1, dtype=int)

    for k, id in enumerate(node_id): node2idx[id] = k

    # cells is element_type -> node_id 
    cells = {}
    cell_data = {}
    el2idx = {}
    for ndim in range(3):
        if (ndim == 1 and finished_lines is not None
            or
            ndim == 2 and finished_faces is not None):
            finished = finished_faces if ndim==2 else finished_lines

            #print("here we are removing unfinished lines",  dimtags, finished_lines)
            xxx = [model.mesh.getElements(ndim, l) for l in finished]
            tmp = {}
            elementTypes = sum([x[0] for x in xxx], [])
            elementTags =sum([x[1] for x in xxx], [])
            nodeTags = sum([x[2] for x in xxx], [])
            dd1 = {};
            dd2 = {}
            for k, el_type in enumerate(elementTypes):
                if not el_type in dd1: dd1[el_type] = []
                if not el_type in dd2: dd2[el_type] = []                
                dd1[el_type] = dd1[el_type]+ elementTags[k]
                dd2[el_type] = dd2[el_type]+ nodeTags[k]
            elementTypes = dd1.keys()
            elementTags = [dd1[k] for k in elementTypes]
            nodeTags = [dd2[k] for k in elementTypes]
        else:
            elementTypes, elementTags, nodeTags = model.mesh.getElements(ndim)            
        for k, el_type in enumerate(elementTypes):
            el_type_name = gmsh_element_type[el_type]
            data = np.array([node2idx[tag] for tag in nodeTags[k]], dtype=int)
            data = data.reshape(-1, num_nodes_per_cell[el_type_name])
            cells[el_type_name] = data
            tmp = np.zeros(max(elementTags[k])+1, dtype=int)
            for kk, id in enumerate(elementTags[k]): tmp[id] = kk
            el2idx[el_type_name] = tmp
            cell_data[el_type_name] = {'geometrical':
                                       np.zeros(len(elementTags[k]), dtype=int),
                                       'physical':
                                       np.zeros(len(elementTags[k]), dtype=int)}

        dimtags =  model.getEntities(dim=ndim)

        if (ndim == 1 and finished_lines is not None
            or
            ndim == 2 and finished_faces is not None):
            finished = finished_faces if ndim==2 else finished_lines
            #print("here we are removing unfinished lines",  dimtags, finished_lines)
            dimtags = [dt for dt in dimtags if dt[1] in finished]
        for dim, tag in dimtags:
            elType2, elTag2, nodeTag2 = model.mesh.getElements(dim=dim,
                                                               tag=tag)
            for k, el_type in enumerate(elType2):                       
                el_type_name = gmsh_element_type[el_type]
                for elTag in elTag2[k]:
                   idx = el2idx[el_type_name][elTag]
                   cell_data[el_type_name]['geometrical'][idx] = tag

        dimtags = model.getPhysicalGroups(dim=dim)
        for dim, ptag in dimtags:
            etags = model.getEntitiesForPhysicalGroup(dim=dim, tag=ptag)
            for etag in etags:
                elType2, elTag2, nodeTag2 = model.mesh.getElements(dim=dim,
                                                                        tag=etag)
                for k, el_type in enumerate(elType2):                       
                    el_type_name = gmsh_element_type[el_type]
                    for elTag in elTag2[k]:
                        idx = el2idx[el_type_name][elTag]
                        cell_data[el_type_name]['physical'][idx] = ptag


    return points, cells, cell_data