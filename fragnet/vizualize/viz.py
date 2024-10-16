import sys
# sys.path.append('../fragnet_edge')
from fragnet.dataset.dataset import FinetuneData
import pandas as pd
from fragnet.dataset.utils import extract_data
import argparse
from omegaconf import OmegaConf
from fragnet.train.utils import TrainerFineTune as Trainer
from fragnet.dataset.data import collate_fn
from torch.utils.data import DataLoader
import torch
from fragnet.dataset.fragments import get_3Dcoords
from fragnet.dataset.fragments import FragmentedMol
from collections import defaultdict
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image
import io
import matplotlib.cm as cm
from fragnet.vizualize.model import FragNetFineTuneViz
import matplotlib.colors as colors
from rdkit import Chem
from rdkit.Chem import rdDepictor
rdDepictor.SetPreferCoordGen(True)
from rdkit import Geometry
import numpy as np
# import config
# import os
import torch
from fragnet.model.cdrp.model import CDRPModel
from fragnet.dataset.fragments import get_3Dcoords
from fragnet.dataset.data import CreateDataCDRP
from fragnet.dataset.data import collate_fn_cdrp
import copy
import torch.nn as nn
from fragnet.vizualize.model import FragNetPreTrainViz, FragNetViz

# MODEL_CONFIG = config.MODEL_CONFIG
# MODEL_PATH = config.MODEL_PATH

WIDTH = 400
HEIGHT = 300


def get_atoms_in_frags(frag):
    
    atoms_in_frags={i:[] for i in range(len(frag.fragments))}
    for i in range(len(frag.fragments)):
        c_frag = frag.fragments[i]
        assert i == c_frag.FragIdx
        for j in c_frag.atom_indices:
            # highlightatoms[j].append(colors[i])
            # highlightatoms[j].append(cm.Reds(fweights[i]))
            # atoms_in_frags[i].append(j)
            atoms_in_frags[c_frag.FragIdx].append(j)

    return atoms_in_frags

def add_atom_numbers(mol):
        
    for atom in mol.GetAtoms():
        # For each atom, set the property "atomNote" to a index+1 of the atom
        atom.SetProp("atomNote", str(atom.GetIdx()))


def highlight_frags(smiles, frag_atoms):
    # smiles = test_dataset[1].smiles
    # mol = Chem.MolFromSmiles(test_dataset[data_id].smiles)
    # mol = Chem.MolFromSmiles(smiles)
    mol = get_3Dcoords(smiles)
    conf = mol.GetConformer(id=0)
    frag = FragmentedMol(mol, conf)
    mol = frag.mol
    add_atom_numbers(mol)
    mol.RemoveAllConformers()
        
    for m in [mol]:
        rdDepictor.Compute2DCoords(m)
    
    for m in [mol]:
        for atom in m.GetAtoms():
            atom.SetIntProp("SourceAtomIdx",atom.GetIdx())
    
    sourceIdxProperty='SourceAtomIdx'
    
    # brics = list(BRICSDecompose(mol, returnMols=True))
    # brics = list(BRICSDecompose(mol, returnMols=True))
    # row = {'c1':brics[2]}
    # row = {'c1':frags[2]}
    # core = qcore
    
    # copy the molecule and core
    mol = Chem.Mol(mol)
    tmol= mol
    
    
    # "Tol" colormap from https://davidmathlogic.com/colorblind
    colors = [(51,34,136),(17,119,51),(68,170,153),(136,204,238),(221,204,119),(204,102,119),(170,68,153),(136,34,85)]
    # "IBM" colormap from https://davidmathlogic.com/colorblind
    colors = [(100,143,255),(120,94,240),(220,38,127),(254,97,0),(255,176,0)]
    # Okabe_Ito colormap from https://jfly.uni-koeln.de/color/
    colors = [(230,159,0),(86,180,233),(0,158,115),(240,228,66),(0,114,178),(213,94,0),(204,121,167)]

    # colors = cm.rainbow(np.linspace(0, 1, 15))
    colors = cm.tab20c(np.linspace(0, 1, 50))
    # colors = cm.cool(np.linspace(0, 1, 15))
    # for i,x in enumerate(colors):
    #     colors[i] = tuple(y/255 for y in x)

    
    
    legend=''
    
    fillRings=True
    width=WIDTH
    height=HEIGHT
    
    #----------------------
    # Identify and store which atoms, bonds, and rings we'll be highlighting
    highlightatoms = defaultdict(list)
    highlightbonds = defaultdict(list)
    atomrads = {}
    widthmults = {}
    
    rings = []
    
    d2d = rdMolDraw2D.MolDraw2DCairo(width,height)
    dos = d2d.drawOptions()
    dos.useBWAtomPalette()
    dos.minFontSize = 10

    # dopts = d2d.drawOptions()
    dos.highlightRadius = .15
    # highlightatoms=[]
    # highlightbonds=[]
    for i in range(len(frag.fragments)):
        for j in frag.fragments[i].atom_indices:
            # highlightatoms[j].append(colors[i])
            # highlightatoms[j].append(cm.Reds(fweights[i]))
            highlightatoms[j].append( tuple(colors[i]) )

            if j in frag_atoms:
                atomrads[j] = .5
            else:
                atomrads[j] = .001
                
                
            
        for j in frag.fragments[i].bond_indices:
            # highlightbonds[j].append(cm.Reds(fweights[i]))
            highlightbonds[j].append( tuple(colors[i]) )
            # widthmults[j] = 20
    
    # highlightatoms = {i:[colors[5]] for i in frag.fragments[1].atom_indices}
    # highlightbonds = {i:[colors[5]] for i in frag.fragments[1].bond_indices}
    
    #----------------------
    # if we are filling rings, go ahead and do that first so that we draw
    # the molecule on top of the filled rings
    if fillRings and rings:
        # a hack to set the molecule scale
        d2d.DrawMoleculeWithHighlights(tmol,legend,dict(highlightatoms),
                                       dict(highlightbonds),
                                       atomrads,widthmults)
        d2d.ClearDrawing()
        conf = tmol.GetConformer()
        for (aring,color) in rings:
            ps = []
            for aidx in aring:
                pos = Geometry.Point2D(conf.GetAtomPosition(aidx))
                ps.append(pos)
            d2d.SetFillPolys(True)
            d2d.SetColour(color)
            d2d.DrawPolygon(ps)
        dos.clearBackground = False
    

    d2d.DrawMoleculeWithHighlights(mol=tmol,legend=legend, highlight_atom_map= dict(highlightatoms),
                                   highlight_bond_map=dict(highlightbonds),
                                   # highlight_bond_map={},
                                   highlight_radii=atomrads,highlight_linewidth_multipliers=widthmults)
    d2d.FinishDrawing()
    bio = io.BytesIO(d2d.GetDrawingText())
    png = Image.open(bio)
    
    # png = IPImage(png)
    return png, mol




from fragnet.dataset.data import get_bond_pair_fbond_graph
def highlight_frag_attention(smiles, fweights, frag_atoms, title=""):

    mol = get_3Dcoords(smiles)
    conf = mol.GetConformer(id=0)
    frag = FragmentedMol(mol, conf)
    mol = frag.mol
    add_atom_numbers(mol)
    mol.RemoveAllConformers()
        
    for m in [mol]:
        rdDepictor.Compute2DCoords(m)
    
    for m in [mol]:
        for atom in m.GetAtoms():
            atom.SetIntProp("SourceAtomIdx",atom.GetIdx())
    
    sourceIdxProperty='SourceAtomIdx'
    

    mol = Chem.Mol(mol)
    tmol= mol
    
    
    # "Tol" colormap from https://davidmathlogic.com/colorblind
    colors = [(51,34,136),(17,119,51),(68,170,153),(136,204,238),(221,204,119),(204,102,119),(170,68,153),(136,34,85)]
    # "IBM" colormap from https://davidmathlogic.com/colorblind
    colors = [(100,143,255),(120,94,240),(220,38,127),(254,97,0),(255,176,0)]
    # Okabe_Ito colormap from https://jfly.uni-koeln.de/color/
    colors = [(230,159,0),(86,180,233),(0,158,115),(240,228,66),(0,114,178),(213,94,0),(204,121,167)]

    colors = cm.rainbow(np.linspace(0, 1, 8))
    # for i,x in enumerate(colors):
    #     colors[i] = tuple(y/255 for y in x)

    
    
    
    fillRings=True
    # width=550
    # height=400
    
    width=WIDTH
    height=HEIGHT
    
    #----------------------
    # Identify and store which atoms, bonds, and rings we'll be highlighting
    highlightatoms = defaultdict(list)
    highlightbonds = defaultdict(list)
    atomrads = {}
    widthmults = {}
    
    rings = []
    
    d2d = rdMolDraw2D.MolDraw2DCairo(width,height)
    dos = d2d.drawOptions()
    dos.useBWAtomPalette()
    dos.minFontSize = 15

    # dopts = d2d.drawOptions()
    dos.highlightRadius = .15
    # highlightatoms=[]
    # highlightbonds=[]
    for i in range(len(frag.fragments)):
        for j in frag.fragments[i].atom_indices:
            # highlightatoms[j].append(colors[i])
            # highlightatoms[j].append(cm.Blues(fweights[i]))
            highlightatoms[j].append(cm.Reds(fweights[i]))
            # highlightatoms[j].append( tuple(colors[i]) )

            # if j in frag_atoms:
            atomrads[j] = .15
            # else:
            #     atomrads[j] = .001
                
                
            
        for j in frag.fragments[i].bond_indices:
            highlightbonds[j].append(cm.Reds(fweights[i]))
            # highlightbonds[j].append(cm.Blues(fweights[i]))
            # highlightbonds[j].append( tuple(colors[i]) )
            # widthmults[j] = 20
    
    # highlightatoms = {i:[colors[5]] for i in frag.fragments[1].atom_indices}
    # highlightbonds = {i:[colors[5]] for i in frag.fragments[1].bond_indices}
    
    #----------------------
    # if we are filling rings, go ahead and do that first so that we draw
    # the molecule on top of the filled rings
    if fillRings and rings:
        # a hack to set the molecule scale
        d2d.DrawMoleculeWithHighlights(tmol,title,dict(highlightatoms),
                                       dict(highlightbonds),
                                       atomrads,widthmults)
        d2d.ClearDrawing()
        conf = tmol.GetConformer()
        for (aring,color) in rings:
            ps = []
            for aidx in aring:
                pos = Geometry.Point2D(conf.GetAtomPosition(aidx))
                ps.append(pos)
            d2d.SetFillPolys(True)
            d2d.SetColour(color)
            d2d.DrawPolygon(ps)
        dos.clearBackground = False
    

    d2d.DrawMoleculeWithHighlights(mol=tmol,legend=title, highlight_atom_map= dict(highlightatoms),
                                   highlight_bond_map=dict(highlightbonds),
                                   # highlight_bond_map={},
                                   highlight_radii=atomrads,highlight_linewidth_multipliers=widthmults)
    d2d.FinishDrawing()
    bio = io.BytesIO(d2d.GetDrawingText())
    png = Image.open(bio)
    
    # png = IPImage(png)
    return png, mol

def get_fragbond_viz(frag_idx, cnx_attr):

    keys = frag_idx.T.numpy().tolist()

    # cnx_attr
    cnx_attr_list = cnx_attr.numpy().tolist()
    # cnx_attr_list
    frag_bond_cnx_attr = { tuple(keys[i]) : cnx_attr_list[i] for i in range(len(keys)) }
    # frag_bond_cnx_attr


    idx_fragbond_index = {}
    node_feautures_fragbond_graph = []
    for ie, i in enumerate(range(0,len(frag_idx[0]), 1)):
    
        id1 = frag_idx[0][i].item()
        id2 = frag_idx[1][i].item()
        idx_fragbond_index[ie] = [id1, id2]
    
        # bond = graph.mol.GetBondBetweenAtoms(id1, id2)
        frag_feature = frag_bond_cnx_attr[ (id1, id2) ]
        node_feautures_fragbond_graph.append(frag_feature)


    node_feautures_fragbond_graph = torch.tensor(node_feautures_fragbond_graph, dtype=torch.float)
    fragbond_index_to_id = {tuple(v):i for i,v in idx_fragbond_index.items()}
    edge_index_fragbond_graph = get_bond_pair_fbond_graph(idx_fragbond_index)

    # print("edge_index_fragbond_graph ", edge_index_fragbond_graph)


    edge_attr_fragbond =[]
    for j in range(len(edge_index_fragbond_graph[0])):

        # print('j: ', j)
        node1 = edge_index_fragbond_graph[0][j] # node ids for the frag graph
        node2 = edge_index_fragbond_graph[1][j]
    
    
        node_1_bond = idx_fragbond_index[ node1 ]
        node_2_bond = idx_fragbond_index[ node2 ]
    
        edge_feature = list( np.array(frag_bond_cnx_attr[tuple( node_1_bond  )]) + np.array(frag_bond_cnx_attr[tuple( node_2_bond )]) )
        
        edge_attr_fragbond.append( edge_feature )


    edge_index_fragbond_graph = torch.tensor(edge_index_fragbond_graph, dtype=torch.int32)
    edge_attr_fragbond = torch.tensor(edge_attr_fragbond, dtype=torch.float)


    return idx_fragbond_index, fragbond_index_to_id, node_feautures_fragbond_graph, edge_index_fragbond_graph, edge_attr_fragbond



def get_regbond_ids_for_fragbond_ids(id1, id2, graph):
    
    frags = graph.fragments
    curr_frag = frags[id1]
    
    for cns in curr_frag.connections:
        # try:
            if cns.EndFragIdx == id2 or cns.BeginFragIdx==id2:
                connection = cns
                break
        # except:
        #     continue
    
    # connection.EndFragIdx
    
    cnx_bond_id = connection.bond_id

    if cnx_bond_id != None:
        
        bond = graph.mol.GetBondWithIdx(cnx_bond_id)
        bid1 = bond.GetEndAtomIdx()
        bid2 = bond.GetBeginAtomIdx()

    elif cnx_bond_id == None:
        bid1 = curr_frag.atom_indices[0] # choose arbitrary two atoms to connect the 
        bid2 = frags[id2].atom_indices[0] # two fragments between which a regular bond
                                            # does not exist
    return bid1, bid2



def df_frag_weights(batch, summed_attn_weights_fbonds):
    
    idx_fragbond_index, fragbond_index_to_id, node_feautures_fragbond_graph, edge_index_fragbond_graph, edge_attr_fragbond = get_fragbond_viz(batch['frag_index'], batch['cnx_attr'])
    
    dfw = pd.DataFrame(zip(idx_fragbond_index.keys(), idx_fragbond_index.values(), summed_attn_weights_fbonds.numpy()),
                 columns=['fbid', 'connection', 'w'])
    
    

    
    for i in dfw.index:
        c = dfw.loc[i, 'connection']
        dfw.loc[i, 'cnx_sorted'] = str(sorted(c))
    
    # dfw.head()
    dfw2 = pd.DataFrame(dfw.groupby('cnx_sorted')['w'].sum(0))

    return dfw, dfw2

import ast

def get_frags(smiles):
    mol = get_3Dcoords(smiles)
    conf = mol.GetConformer(id=0)
    graph = FragmentedMol(mol, conf)
    frags = graph.fragments
    return graph, frags

def min_max(arr):
    arr = arr.numpy()
    return (arr-np.min(arr))/(np.max(arr) - np.min(arr) )

def get_bond_weights_from_frag_graph(dfw2, graph):
    
    bond_weights_from_frag_graph ={}
    for i in dfw2.index:
    
        fid1, fid2  = ast.literal_eval(i)
    
        # break
        # try:
            
        bid1, bid2 = get_regbond_ids_for_fragbond_ids(fid1, fid2, graph)
        
        w = dfw2.loc[i, 'w']
    
        bond_weights_from_frag_graph[tuple((bid1, bid2))] = sum(w)
        # print("s", fid1, fid2, bid1, bid2)
        # except:
        #     print(fid1, fid2, bid1, bid2)
        #     pass
    
    frag_atoms = [list(i) for i in list(bond_weights_from_frag_graph.keys())]
    
    frag_atoms = list(set(sum(frag_atoms, [])))


    return bond_weights_from_frag_graph, frag_atoms


class FragNetFineTuneBase(nn.Module):
    
    def __init__(self, n_classes=1, atom_features=167, frag_features=167, edge_features=17, 
                 num_layer=4, num_heads=4, drop_ratio=0.15,
                h1=256, h2=256, h3=256, h4=256, act='celu',emb_dim=128, fthead='FTHead3'):
        super().__init__()
        

        self.pretrain = FragNet(num_layer=num_layer, drop_ratio=drop_ratio, 
                                num_heads=num_heads, emb_dim=emb_dim,
                                atom_features=atom_features, frag_features=frag_features, edge_features=edge_features)


        if fthead == 'FTHead1':
            self.fthead = FTHead1(n_classes=n_classes)
        elif fthead == 'FTHead2':
            print('using FTHead2' )
            self.fthead = FTHead2(n_classes=n_classes)
        elif fthead == 'FTHead3':
            print('using FTHead3' )
            self.fthead = FTHead3(n_classes=n_classes,
                             h1=h1, h2=h2, h3=h3, h4=h4,
                             drop_ratio=drop_ratio, act=act)
            
        elif fthead == 'FTHead4':
            print('using FTHead4' )
            self.fthead = FTHead4(n_classes=n_classes,
                             h1=h1, drop_ratio=drop_ratio, act=act)
        
                    
        
    def forward(self, batch):
        
        x_atoms, x_frags, x_edge, x_fedge = self.pretrain(batch)
        
        x_frags_pooled = scatter_add(src=x_frags, index=batch['frag_batch'], dim=0)
        x_atoms_pooled = scatter_add(src=x_atoms, index=batch['batch'], dim=0)
        
        cat = torch.cat((x_atoms_pooled, x_frags_pooled), 1)

            
        return cat



def load_partial_weights(model_viz, pretrained_dict):
    
    model_dict = model_viz.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict) 
    
    # 3. load the new state dict
    model_viz.load_state_dict(model_dict)

    return model_viz
    
class FragNetVizApp:

    def __init__(self, model_config, model_checkpoint, model_type='property', device='cpu'):

        # parser = argparse.ArgumentParser()
 
        # parser.add_argument('--config', help="configuration file *.yml", type=str, required=False, 
        #                     default='../fragnet_edge/exps/ft/pnnl_set2/fragnet_hpdl_exp1s_pt4_10/config_exp100.yaml')
        # parser.add_argument('--config', help="configuration file *.yml", type=str, required=False, 
        #                     default='../fragnet_edge/exps/ft/pnnl_set2/exp1s_h4pt4.yaml')
        
        # parser.add_argument('--config', help="configuration file *.yml", type=str, required=False, 
        #                     default=MODEL_CONFIG)
 

        
        # args = parser.parse_args('')

        # if args.config:  # args priority is higher than yaml
        opt = OmegaConf.load(model_config)
        OmegaConf.resolve(opt)
        # opt.update(vars(args))
        args=opt

        if model_type in ['property', 'cdrp']:
            model = FragNetFineTuneViz(n_classes=args.finetune.model.n_classes, 
                                        atom_features=args.atom_features, 
                                        frag_features=args.frag_features, 
                                        edge_features=args.edge_features,
                                        num_layer=args.finetune.model.num_layer, 
                                        drop_ratio=args.finetune.model.drop_ratio,
                                            num_heads=args.finetune.model.num_heads, 
                                            emb_dim=args.finetune.model.emb_dim,
                                            h1=args.finetune.model.h1,
                                            h2=args.finetune.model.h2,
                                            h3=args.finetune.model.h3,
                                            h4=args.finetune.model.h4,
                                            act=args.finetune.model.act,
                                            fthead=args.finetune.model.fthead)

            if model_type == 'cdrp':
                model = CDRPModel(model, args.gene_dim, device)
            
        elif model_type=='energy':
            model = FragNetPreTrainViz(num_layer=args.pretrain.num_layer,
                        drop_ratio=args.pretrain.drop_ratio,
                            num_heads=args.pretrain.num_heads,
                            emb_dim=args.pretrain.emb_dim,
                            atom_features=args.atom_features,
                            frag_features=args.frag_features,
                            edge_features=args.edge_features)


            
        model.load_state_dict(torch.load(model_checkpoint,
                                         map_location=torch.device('cpu')))
        

        self.model = model
        self.dataset = FinetuneData(target_name='log_sol', data_type='exp1s')
        self.viz_model = FragNetViz(num_layer=args.finetune.model.num_layer, drop_ratio=args.finetune.model.drop_ratio, 
                                num_heads=args.finetune.model.num_heads, emb_dim=args.finetune.model.emb_dim,
                                atom_features=args.atom_features, frag_features=args.frag_features, edge_features=args.edge_features)

        self.viz_model = load_partial_weights(self.viz_model, torch.load(model_checkpoint, map_location=torch.device('cpu')) )
    
        # self.smiles = smiles
        
        # self.get_weights(smiles)


    
    def calc_weights(self, smiles):
        self.smiles = smiles
        self.mol = self.get_mol(self.smiles)

        train = pd.DataFrame.from_dict({'smiles': [smiles], 'log_sol': [1.]})
        ds = self.dataset.get_ft_dataset(train)
        ds = extract_data(ds)

        test_loader = DataLoader(ds, collate_fn=collate_fn, batch_size=1, shuffle=False, drop_last=False)

        batch = next(iter(test_loader))


        with torch.no_grad():
            self.model.eval()
            # if pretrain model is used
            # x_atoms, x_frags, x_edge, x_fedge, summed_attn_weights_atoms, summed_attn_weights_frags, \
            #         summed_attn_weights_bonds, summed_attn_weights_fbonds = self.model.pretrain(batch)

            prop_prediction, summed_attn_weights_atoms, summed_attn_weights_frags, \
                    summed_attn_weights_bonds, summed_attn_weights_fbonds = self.model(batch)

            # x_atoms, x_frags, x_edge, x_fedge, summed_attn_weights_atoms, summed_attn_weights_frags, \
            #         summed_attn_weights_bonds, summed_attn_weights_fbonds = self.viz_model(batch)

            prop_prediction = prop_prediction.item()

        self.summed_attn_weights_atoms = summed_attn_weights_atoms
        self.summed_attn_weights_frags = summed_attn_weights_frags
        self.summed_attn_weights_bonds = summed_attn_weights_bonds
        self.summed_attn_weights_fbonds = summed_attn_weights_fbonds
        self.batch = batch

        a1 = summed_attn_weights_bonds[::2]
        a2 = summed_attn_weights_bonds[1::2]
        bond_weights = a1+a2/2
        bond_weights = bond_weights.sum(1)

        self.bond_weights_sc = min_max(bond_weights)

        return prop_prediction

    def calc_weights_cdrp(self, smiles, gene_expr):

        self.smiles = smiles
        self.mol = get_3Dcoords(smiles)
        conf = self.mol.GetConformer(id=0)
        x = [smiles, [-1], self.mol, conf, gene_expr]
        
        create_data = CreateDataCDRP(data_type='exp1s', create_bond_graph_data=True, 
                                            add_dhangles=False)
        
        ds = create_data.create_data_point(x)
        test_loader = DataLoader([ds], collate_fn=collate_fn_cdrp, batch_size=1, shuffle=False, drop_last=False)


        batch = next(iter(test_loader))

        with torch.no_grad():
            self.model.eval()
            x_atoms, x_frags, x_edge, x_fedge, summed_attn_weights_atoms, summed_attn_weights_frags, \
                    summed_attn_weights_bonds, summed_attn_weights_fbonds = self.model.drug_model.pretrain(batch)
            

        self.summed_attn_weights_atoms = summed_attn_weights_atoms
        self.summed_attn_weights_frags = summed_attn_weights_frags
        self.summed_attn_weights_bonds = summed_attn_weights_bonds
        self.summed_attn_weights_fbonds = summed_attn_weights_fbonds
        self.batch = batch

        a1 = summed_attn_weights_bonds[::2]
        a2 = summed_attn_weights_bonds[1::2]
        bond_weights = a1+a2/2
        bond_weights = bond_weights.sum(1)

        self.bond_weights_sc = min_max(bond_weights)




    
    def vizualize_atom_weights(self, hide_bond_weights, hide_atom_weights):
        atom_weights = self.summed_attn_weights_atoms.sum(1)
        png = self.atom_highlight(self.mol, atom_weights, hide_bond_weights, hide_atom_weights)

        return png, atom_weights



    def get_mol(self, smiles):


        mol = get_3Dcoords(smiles)
        conf = mol.GetConformer(id=0)
        frag = FragmentedMol(mol, conf)
        mol = frag.mol
        # mol.RemoveAllConformers()
        # self.mol = mol
        return mol


# def highlight_atoms_weights(atom_weights, bond_weights, title=""):
    
#     highlightatoms = defaultdict(list)
#     highlightbonds = defaultdict(list)
#     bond_widths = defaultdict(list)
#     atom_ids=[]
#     arads = {}

#     if atom_weights is not None:
#         for i in range(len(atom_weights)):
        
#             highlightatoms[i].append(cm.Reds(atom_weights[i]))
#             atom_ids.append(i)
#             arads[i] = 0.2
        


#     if bond_weights is not None:
#         for i in range(len(bond_weights)):
#             highlightbonds[i].append(cm.Reds(bond_weights[i]))
#             bond_widths[i].append(1)
#     # else:
#     #     highlightbonds={}
        
        
#     d2d = rdMolDraw2D.MolDraw2DCairo(width=400, height=300)
#     d2d.DrawMoleculeWithHighlights(mol=mol,legend=title, highlight_atom_map=dict(highlightatoms),
#                                    highlight_bond_map=dict(highlightbonds),
#                                    # highlight_bond_map=dict(highlightbonds),
#                                    highlight_radii={},highlight_linewidth_multipliers={} )
    
#     d2d.FinishDrawing()
#     bio = io.BytesIO(d2d.GetDrawingText())
#     png = Image.open(bio)

#     return png, highlightatoms, highlightbonds


    def atom_highlight(self, mol, atom_weights, hide_bond_weights=False, hide_atom_weights=False):
        # atom_weights = atom_weights/sum(atom_weights)
        # cmap = colors.LinearSegmentedColormap.from_list('nameofcolormap',['g','r'],gamma=2.0)

        mol_copy = copy.deepcopy(mol)
        mol_copy.RemoveAllConformers()

        
        add_atom_numbers(mol_copy)
        for m in [mol_copy]:
            for atom in m.GetAtoms():
                atom.SetIntProp("SourceAtomIdx",atom.GetIdx())
            


        
        atom_weights = atom_weights.numpy()
        atom_weights = ( atom_weights - np.min(atom_weights) ) /  (np.max(atom_weights) - np.min(atom_weights) )
        highlightatoms = defaultdict(list)
        highlightbonds = defaultdict(list)
        bond_widths = defaultdict(list)
        atom_ids=[]
        arads = {}
        # for a in mol.GetAtoms():

        for i in range(len(atom_weights)):
            # for j in frag.fragments[i].atom_indices:
                # highlightatoms[j].append(colors[i])
            # highlightatoms[i].append(cm.coolwarm(atom_weights[i]))
            highlightatoms[i].append(cm.Reds(atom_weights[i]))
            # highlightatoms[i].append(cm.GnBu(atom_weights[i]))
            # highlightatoms[i].append(cm.coolwarm(atom_weights[i]))
            atom_ids.append(i)
            arads[i] = 0.3




        # if bond_weights is not None:
        if hide_bond_weights:
            highlight_bond_map = {}

        else:
            for i in range(len(self.bond_weights_sc)):
                highlightbonds[i].append(cm.Reds(self.bond_weights_sc[i]))
                bond_widths[i].append(1)
            highlight_bond_map=dict(highlightbonds)
                
        if hide_atom_weights:
            highlight_atom_map={}
        else:
            highlight_atom_map=dict(highlightatoms)

        d2d = rdMolDraw2D.MolDraw2DCairo(width=WIDTH, height=HEIGHT)
        d2d.DrawMoleculeWithHighlights(mol=mol_copy,legend="", highlight_atom_map=highlight_atom_map,
                                    # highlight_bond_map={},
                                    highlight_bond_map=highlight_bond_map,
                                    highlight_radii={},highlight_linewidth_multipliers={})

        d2d.FinishDrawing()
        bio = io.BytesIO(d2d.GetDrawingText())
        png = Image.open(bio)

        return png
    


    def frag_weight_highlight(self):

        frag_weights = self.summed_attn_weights_frags.sum(1).detach().numpy()
        frag_weights_sc = (frag_weights - min(frag_weights))/(max(frag_weights) - min(frag_weights))

        dfw, dfw2 = df_frag_weights(self.batch, self.summed_attn_weights_fbonds)
        graph, frags = get_frags(self.smiles)

        # get the atoms that belong to each fragment
        atoms_in_frags = get_atoms_in_frags(graph)

        bond_weights_from_frag_graph, frag_atoms = get_bond_weights_from_frag_graph(dfw2, graph)

        # fragment weights
        dffw = pd.DataFrame(np.column_stack([[i.FragIdx for i in frags], frag_weights ]))
        dffw.columns = ['fragment', 'weight']
        dffw['smiles'] = self.smiles
        # data2[smiles] =dffw
        # fragment weights

        # connection weights
        dfw2.loc[:, 'sum_weight'] = [dfw2.iloc[i]['w'].sum() for  i in range(dfw2.shape[0])]
        dfw2['smiles'] = self.smiles
        dfw2.reset_index(inplace=True)
        dfw2.rename(columns={'cnx_sorted': 'connection', 'sum_weight':'weight'}, inplace=True)
        dfw2 = dfw2.loc[:, ['connection', 'weight', 'smiles']]
        # dfw2 = dfw2.loc[:, ['connection', 'weight', 'smiles']]
        # data[smiles] = dfw2
        # connection weights

        # png, mol = highlight_frags(smiles, frag_weights, frag_atoms)
        # png, mol = highlight_atoms(self.smiles, frag_weights_sc, frag_atoms, title='Fragment Weights')
        png_frag_attn, mol = highlight_frag_attention(self.smiles, frag_weights_sc, frag_atoms, title='')


        png_frag_highlight, _ = highlight_frags(self.smiles, frag_atoms)



        

        return png_frag_attn, png_frag_highlight, dffw, dfw2, atoms_in_frags
        # return png_frag