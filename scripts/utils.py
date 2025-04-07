import io
import requests
from PIL import Image
from openai import OpenAI

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from pubchempy import get_compounds

from rdkit.Chem import rdDepictor
rdDepictor.SetPreferCoordGen(True)

class Agent():

    def __init__(self, API_URL):

        self.client = OpenAI(
            api_key="EMPTY",
            base_url=API_URL,
        )

        print (f"[API Ready] {API_URL}")
    
    def chat(self, img_path, query="What is the SMILES of the molecule in the image?", temperature=0.0):

        
        chat_response = self.client.chat.completions.create(
            model="Qwen2-7B-Instruct",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": img_path
                            },
                        },
                        {"type": "text", "text": query},
                    ],
                },
            ],
            temperature=temperature
        )
        return chat_response.choices[0].message.content


class Visualizer():

    def __init__(self, color_map=None):

        if not color_map:
            self.color_map = [(0.8, 0.0, 0.8), (0.8 , 0.8, 0), (0, 0.8, 0.8), (0, 0, 0.8), (0.8, 0.0, 0.0), (0.0, 0.8, 0.0), (0.4, 0.0, 0.4), (0.4 , 0.4, 0), (0, 0.4, 0.4), (0, 0, 0.4), (0.4, 0.0, 0.0), (0.0, 0.4, 0.0)]
        else:
            self.color_map = color_map

    def _add_colors_to_map(self, items, colors, color_idx):
        for item in items:
            if item not in colors:
                colors[item] = []
            if self.color_map[color_idx] not in colors[item]:
                colors[item].append(self.color_map[color_idx])


    def draw_mol(self, smiles, filename, size=[1200,1200], highlight_patts=None, is_SMART=None):
        """
        可视化分子结构图（黑白）
        :param smiles: str, SMILES string
        :param filename: str
        :param size: list(int, int)
        :param highlight_patts: list(str), SMILES string
        """
        mol = Chem.MolFromSmiles(smiles)
        drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
        opts = drawer.drawOptions()
        opts.useBWAtomPalette()
        opts.fillHighlights = False

        if highlight_patts:
            color_atom = dict()
            color_bond = dict()
            for idx, highlight_patt in enumerate(highlight_patts):
                if is_SMART[idx]:
                    highlight_patt = Chem.MolFromSmarts(highlight_patt)
                else:
                    highlight_patt = Chem.MolFromSmiles(highlight_patt)
                matches = mol.GetSubstructMatches(highlight_patt)

                hit_at_group = list()
                for i in matches:
                    hit_at_group.append(list(i))
                    self._add_colors_to_map(list(i), color_atom, idx%len(self.color_map))

                hit_bond_group = list()
                for at_group in hit_at_group:
                    for bond in highlight_patt.GetBonds():
                        aid1 = at_group[bond.GetBeginAtomIdx()]
                        aid2 = at_group[bond.GetEndAtomIdx()]
                        hit_bond_group.append(mol.GetBondBetweenAtoms(aid1, aid2).GetIdx())
                self._add_colors_to_map(hit_bond_group, color_bond, idx%len(self.color_map))

            drawer.DrawMoleculeWithHighlights(mol, legend="", highlight_atom_map=color_atom, highlight_bond_map=color_bond, highlight_radii={}, highlight_linewidth_multipliers={})
        else:
            drawer.DrawMolecule(mol)

        drawer.FinishDrawing()

        pil_image = Image.open(io.BytesIO(drawer.GetDrawingText()))
        pil_image.save(filename)


def smiles2iupac(smi):
    """
    Get Iupac from SMILES
    :param smi: SMILES
    :return iupac
    :return err infos
    """
    # Rewrite SMILES
    try:
        smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi), isomericSmiles=True, kekuleSmiles=True)
    except Exception:
        return None, "Invalid SMILES string"
    
    # Query the PubChem database
    # 1. with SMILES
    r = requests.get(
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/"
        + smi
        + "/property/IUPACName/JSON"
    )
    # 2. 应对smiles中存在“/”的，如“CC/C(=C\CC/C(=C/C(=O)OC)/C)/CC[C@@H]1[C@](O1)(C)CC”
    if r.status_code != 200:
        # SMILES 2 CID
        try:
            comps = get_compounds(smi, 'smiles')
            cid = str(comps[0].cid)
        except:
            return None, "HTTP related"
        # Query with CID
        if cid:
            r = requests.get(
                "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
                + cid
                + "/property/IUPACName/JSON"
            )
            if r.status_code != 200:
                return None, "HTTP related"
            data = r.json()
        else:
            return None, "Can not find CID"
    else:
        data = r.json()

    # Get IUPAC name
    try:
        iupac = data["PropertyTable"]["Properties"][0]["IUPACName"]
        return iupac, None
    except KeyError:
        return None, "Can not be converted to IUPAC name"


class FuncGroupsAgent():

    def __init__(self):

        self.dict_fgs_SMART = {
            "furan": "o1cccc1",
            "aldehydes": " [CX3H1](=O)[#6]",
            "esters": " [#6][CX3](=O)[OX2H0][#6]",
            "ketones": " [#6][CX3](=O)[#6]",
            "thiol groups": " [SH]",
            "alcohol groups": " [OH]",
            "methyl amide": "*-[N;D2]-[C;D3](=O)-[C;D1;H3]", # -NC(=O)CH3
            "carboxylic acids": "*-C(=O)[O;D1]", # -C(=O)O
            "carbonyl methyl ester": "*-C(=O)[O;D2]-[C;D1;H3]", #-C(=O)OMe
            "terminal aldehyde": "*-C(=O)-[C;D1]", #-C(=O)H
            "amide": "*-C(=O)-[N;D1]", # -C(=O)N
            "carbonyl methyl": "*-C(=O)-[C;D1;H3]", # -C(=O)CH3
            "isocyanate": "*-[N;D2]=[C;D2]=[O;D1]", # -N=C=O
            "isothiocyanate": "*-[N;D2]=[C;D2]=[S;D1]", # -N=C=S
            "nitro": "*-[N;D3](=[O;D1])[O;D1]", # -NO2
            "nitroso": "*-[N;R0]=[O;D1]", # -N=O
            "oximes": "*=[N;R0]-[O;D1]", # =N-O
            "imines": "*=[N;R0]-[C;D1;H3]", # =NCH3
            "Imines": "*-[N;R0]=[C;D1;H2]", # -N=CH2
            "terminal azo": "*-[N;D2]=[N;D2]-[C;D1;H3]", # -N=NCH3
            "hydrazines": "*-[N;D2]=[N;D1]", # -N=N
            "diazo": "*-[N;D2]#[N;D1]", # -N#N
            "cyano": "*-[C;D2]#[N;D1]", # -C#N
            "primary sulfonamide": "*-[S;D4](=[O;D1])(=[O;D1])-[N;D1]", # -SO2NH2
            "methyl sulfonamide": "*-[N;D2]-[S;D4](=[O;D1])(=[O;D1])-[C;D1;H3]", # -NHSO2CH3
            "sulfonic acid": "*-[S;D4](=O)(=O)-[O;D1]", # -SO3H
            "methyl ester sulfonyl": "*-[S;D4](=O)(=O)-[O;D2]-[C;D1;H3]", # -SO3CH3
            "methyl sulfonyl": "*-[S;D4](=O)(=O)-[C;D1;H3]", # -SO2CH3
            "sulfonyl chloride": "*-[S;D4](=O)(=O)-[Cl]", # -SO2Cl
            "methyl sulfinyl": "*-[S;D3](=O)-[C;D1]", # -SOCH3
            "methyl thio": "*-[S;D2]-[C;D1;H3]", # -SCH3
            "thiols": "*-[S;D1]", # -S
            "thiocarbonyls": "*=[S;D1]", # =S
            "halogens": "*-[#9,#17,#35,#53]", # -X
            "t-butyl": "*-[C;D4]([C;D1])([C;D1])-[C;D1]", # -tBu
            "tri fluoromethyl": "*-[C;D4](F)(F)F", # -CF3
            "acetylenes": "*-[C;D2]#[C;D1;H]", # -C#CH
            "cyclopropyl": "*-[C;D3]1-[C;D2]-[C;D2]1", # -cPropyl
            "ethoxy": "*-[O;D2]-[C;D2]-[C;D1;H3]", # -OEt
            "methoxy": "*-[O;D2]-[C;D1;H3]", # -OMe
            "side-chain hydroxyls": "*-[O;D1]", # -O
            "side-chain aldehydes": "*=[O;D1]", # =O
            "primary amines": "*-[N;D1]", # -N
            "nitriles": "*#[N;D1]", # #N

            "acyl halides": "[C](=[O])[F,Cl,Br,I]", # R-CO-X (X = F, Cl, Br, I)
            "alkenes": "[C]=[C]", #  R-CH=CH-R'
            "amidines": "[C](=[N])[N]", # R-C(=NH)-NH2
            "amino acids": "[NH2][C][C](=[O])[OH]", # H2N-CHR-COOH
            "azides": "[C][N]=[N+]=[N-]", # R-N3
            "diamines": "[NH2][C][C][NH2]", # H2N-R-NH2,
            "epoxides": "[C]1[O][C]1", # 
            "halocarbamates": "[F,Cl,Br,I][C](=[O])[NH][C]", # X-CO-NH-R (X = F, Cl, Br, I)
            "halocarbonates": "[F,Cl,Br,I][C](=[O])[O][C]", # X-CO-OR (X = F, Cl, Br, I)
            "halooximes": "[C](=[N][OH])[F,Cl,Br,I]", # R-C(=NOH)-X (X = F, Cl, Br, I)
            "hydrazides": "[C](=[O])[NH][NH2]", # R-CO-NH-NH2
            "hydrazines": "[C][NH][NH2]", # R-NH-NH2
            "hydrazones": "[C](=[N])[NH2]", # R2C=N-NH2
            "hydroxylamines": "[C][NH][OH]", # R-NH-OH
            "sulfonamides": "[C][S](=[O])(=[O])[NH2]", # R-SO2-NH2
            "thiols": "[C][SH]" # R-SH
        }

        self.dict_fgs_SMILES = {
            "cyclopropane": "C1CC1",
            "aziridine": "C1CN1",
            "oxirane": "C1CO1",
            "azetidin-2-one": "C1CNC1=O",
            "azetidine": "C1CNC1",
            "oxetane": "C1COC1",
            "cyclobutane": "C1CCC1",
            "1,2,4-oxadiazole": "C1=NOC=N1",
            "1,2,4-thiadiazole": "C1=NSC=N1",
            "1,3-thiazole": "C1=CSC=N1",
            "1,4-dihydro-1,2,4-triazol-5-one": "C1=NNC(=O)N1",
            "1,4-dihydropyrazol-5-one": "C1C=NNC1=O",
            "1,3-oxazolidin-2-one": "C1COC(=O)N1",
            "1H-1,2,4-triazole": "C1=NC=NN1",
            "2H-1,2,3-triazole": "C1=NNN=C1",
            "2H-furan-5-one": "C1C=CC(=O)O1",
            "furan-3-one": "C1C(=O)C=CO1",
            "1,3-thiazolidine-2,4-dione": "C1C(=O)NC(=O)S1",
            "2H-tetrazole": "C1=NNN=N1",
            "cyclopentane": "C1CCCC1",
            "cyclopentene": "C1CC=CC1",
            "imidazoline": "C1CN=CN1",
            "isoxazole": "C1=CON=C1",
            "isoxazoline": "C1CON=C1",
            "oxazole": "C1=COC=N1",
            "oxazoline": "C1COC=N1",
            "pyrazole": "C1=CNN=C1",
            "imidazole": "C1=CN=CN1",
            "pyrrole": "C1=CNC=C1",
            "furan": "C1=COC=C1",
            "1,3-thiazolidin-4-one": "C1C(=O)NCS1",
            "pyrrolidin-2-one": "C1CC(=O)NC1",
            "1,2-dihydropyrrol-5-one": "C1C=CC(=O)N1",
            "1,2-dihydropyrazol-3-one": "C1=CNNC1=O",
            "pyrrolidine": "C1CCNC1",
            "2-sulfanylideneimidazolidin-4-one": "C1C(=O)NC(=S)N1",
            "1,3,5-triazine": "C1=NC=NC=N1",
            "1,4-dihydropyridine": "C1C=CNC=C1",
            "1H-pyridazin-6-one": "C1=CC(=O)NN=C1",
            "1H-pyrimidin-6-one": "C1=CN=CNC1=O",
            "1H-pyridin-2-one": "C1=CC(=O)NC=C1",
            "barbituric acid": "C1C(=O)NC(=O)NC1=O",
            "benzene": "C1=CC=CC=C1",
            "cyclohexane": "C1CCCCC1",
            "cyclohexene": "C1CCC=CC1",
            "glutarimide": "C1CC(=O)NC(=O)C1",
            "morpholine": "C1COCCN1",
            "1,3-oxazinane": "C1CNCOC1",
            "thiane 1,1-dioxide": "C1CCS(=O)(=O)CC1",
            "1,4-thiazinane 1,1-dioxide": "C1CS(=O)(=O)CCN1",
            "oxane": "C1CCOCC1",
            "piperazin-2-one": "C1CNC(=O)CN1",
            "piperazine": "C1CNCCN1",
            "piperidin-2-one": "C1CCNC(=O)C1",
            "piperidine": "C1CCNCC1",
            "2-hydroxyoxaborinane": "B1(CCCCO1)O",
            "pyridazine": "C1=CC=NN=C1",
            "pyridine": "C1=CC=NC=C1",
            "pyrimidine": "C1=CN=CN=C1",
            "uracil": "C1=CNC(=O)NC1=O",
            "6-azauracil": "C1=NNC(=O)NC1=O",
            "4H-1,2,4-triazin-5-one": "C1=NN=CNC1=O",
            "azepane": "C1CCCNCC1",
            "1,4-diazepane": "C1CNCCNC1",
            "3-azabicyclo[3.1.0]hexane": "C1C2C1CNC2",
            "bicyclo[3.2.0]hept-2-ene": "C1CC2C1CC=C2",
            "4-thia-1-azabicyclo[3.2.0]heptan-7-one": "C1CSC2N1C(=O)C2",
            "1-azabicyclo[3.2.0]hept-2-en-7-one": "C1C=CN2C1CC2=O",
            "(5R)-4,4-dioxo-4λ6-thia-1-azabicyclo[3.2.0]heptan-7-one": "C1CS(=O)(=O)[C@H]2N1C(=O)C2",
            "1,4,5,6-tetrahydropyrrolo[3,4-c]pyrazole": "C1C2=C(CN1)NN=C2",
            "6,7-dihydro-5H-pyrrolo[1,2-c]imidazole": "C1CC2=CN=CN2C1",
            "1,2,3,3a,4,5,6,6a-octahydrocyclopenta[c]pyrrole": "C1CC2CNCC2C1",
            "2-azabicyclo[3.1.0]hexane": "C1CNC2C1C2",
            "5-thia-1-azabicyclo[4.2.0]oct-2-en-8-one": "C1C=CN2C(S1)CC2=O",
            "1,3-benzodioxole": "C1OC2=CC=CC=C2O1",
            "2,3-dihydro-1,3-benzothiazole 1,1-dioxide": "C1NC2=CC=CC=C2S1(=O)=O",
            "1-hydroxy-3H-2,1-benzoxaborole": "B1(C2=CC=CC=C2CO1)O",
            "1-benzothiophene": "C1=CC=C2C(=C1)C=CS2",
            "1,3-benzoxazole": "C1=CC=C2C(=C1)N=CO2",
            "1,3-dihydroimidazo[4,5-b]pyridin-2-one": "C1=CC2=C(NC(=O)N2)N=C1",
            "7,9-dihydropurin-8-one": "C1=C2C(=NC=N1)NC(=O)N2",
            "1H-pyrazolo[3,4-d]pyrimidine": "C1=C2C=NNC2=NC=N1",
            "1H-pyrrolo[2,3-b]pyridine": "C1=CC2=C(NC=C2)N=C1",
            "1H-thieno[2,3-d]pyrimidine-2,4-dione": "C1=CSC2=C1C(=O)NC(=O)N2",
            "1,6-dihydropyrazolo[4,3-d]pyrimidin-7-one": "C1=NNC2=C1N=CNC2=O",
            "3H-imidazo[5,1-f][1,2,4]triazin-4-one": "C1=C2C(=O)NC=NN2C=N1",
            "3,5-dihydropyrrolo[3,2-d]pyrimidin-4-one": "C1=CNC2=C1N=CNC2=O",
            "2,3-dihydro-1-benzofuran": "C1COC2=CC=CC=C21",
            "isoindoline": "C1C2=CC=CC=C2CN1",
            "2H-triazolo[4,5-b]pyrazine": "C1=NC2=NNN=C2N=C1",
            "4,5,6,7-tetrahydropyrazolo[1,5-a]pyrimidine": "C1CNC2=CC=NN2C1",
            "5,6,7,8-tetrahydro-[1,2,4]triazolo[4,3-a]pyrazine": "C1CN2C=NN=C2CN1",
            "5,6,7,8-tetrahydro-[1,2,4]triazolo[1,5-a]pyrazine": "C1CN2C(=NC=N2)CN1",
            "5,6,7,8-tetrahydroimidazo[1,5-a]pyrazine": "C1CN2C=NC=C2CN1",
            "4,5,6,7-tetrahydrothieno[3,2-c]pyridine": "C1CNCC2=C1SC=C2",
            "1,5,6,7-tetrahydropyrrolo[3,2-c]pyridin-4-one": "C1CNC(=O)C2=C1NC=C2",
            "6,7-dihydro-5H-imidazo[2,1-b][1,3]oxazine": "C1CN2C=CN=C2OC1",
            "7,8-dihydro-6H-pyrrolo[1,2-a]pyrimidin-4-one": "C1CC2=NC=CC(=O)N2C1",
            "7H-pyrrolo[2,3-d]pyrimidine": "C1=CNC2=NC=NC=C21",
            "indole": "C1=CC=C2C(=C1)C=CN2",
            "benzimidazole": "C1=CC=C2C(=C1)NC=N2",
            "benzofuran": "C1=CC=C2C(=C1)C=CO2",
            "1,2-benzothiazole": "C1=CC=C2C(=C1)C=NS2",
            "imidazo[1,2-a]pyridine": "C1=CC2=NC=CN2C=C1",
            "imidazo[1,2-a]pyrimidine": "C1=CN2C=CN=C2N=C1",
            "[1,2,4]triazolo[1,5-a]pyridine": "C1=CC2=NC=NN2C=C1",
            "imidazo[1,2-b][1,2,4]triazine": "C1=CN2C(=N1)N=CC=N2",
            "imidazo[1,2-b]pyridazine": "C1=CC2=NC=CN2N=C1",
            "imidazo[1,5-a]pyrazine": "C1=CN2C=NC=C2C=N1",
            "pyrazolo[1,5-a]pyrazine": "C1=CN2C(=CC=N2)C=N1",
            "1H-pyrazolo[3,4-b]pyridine": "C1=CC2=C(NN=C2)N=C1",
            "indan": "C1CC2=CC=CC=C2C1",
            "indazole": "C1=CC=C2C(=C1)C=NN2",
            "oxindole": "C1C2=CC=CC=C2NC1=O",
            "isoindole-1,3-dione": "C1C2=CC=CC=C2C(=O)N1",
            "2,3-dihydroisoindol-1-one": "C1C2=CC=CC=C2C(=O)N1",
            "pyrazolo[1,5-a]pyrimidine": "C1=CN2C(=CC=N2)N=C1",
            "hypoxanthine": "C1=NC2=C(N1)C(=O)NC=N2",
            "pyrrolo[2,1-f][1,2,4]triazine": "C1=CN2C(=C1)C=NC=N2",
            "purine": "C1=C2C(=NC=N1)N=CN2",
            "xanthine": "C1=NC2=C(N1)C(=O)NC(=O)N2",
            "1,4,5,6-tetrahydropyrazolo[3,4-c]pyridin-7-one": "C1CNC(=O)C2=C1C=NN2",
            "1H-indole-4,7-dione": "C1=CC(=O)C2=C(C1=O)C=CN2",
            "1H-thieno[3,4-d]pyrimidine-2,4-dione": "C1=C2C(=CS1)NC(=O)NC2=O",
            "(4aR,7aS)-2,3,4,4a,5,6,7,7a-octahydro-1H-pyrrolo[3,4-b]pyridine": "C1C[C@@H]2CNC[C@H]2NC1",
            "(3aS,7aR)-2,3,3a,4,5,6,7,7a-octahydro-1H-isoindole": "C1CC[C@H]2CNC[C@H]2C1",
            "(3aS,7aS)-2,3,3a,4,5,6,7,7a-octahydro-1H-isoindole": "C1CC[C@@H]2CNC[C@H]2C1",
            "2,3,4,7,8,8a-hexahydro-1H-pyrrolo[1,2-a]pyrazin-6-one": "C1CC(=O)N2C1CNCC2",
            "1,3,4,6,7,8,9,9a-octahydropyrazino[2,1-c][1,4]oxazine": "C1CN2CCOCC2CN1",
            "1,2,3,4-tetrahydroisoquinoline": "C1CNCC2=CC=CC=C21",
            "3,4-dihydro-2H-chromene": "C1CC2=CC=CC=C2OC1",
            "2-hydroxyquinoline": "C1=CC=C2C(=C1)C=CC(=O)N2",
            "1H-1,8-naphthyridin-2-one": "C1=CC2=C(NC(=O)C=C2)N=C1",
            "4H-pyrido[1,2-a]pyrimidin-4-one": "C1=CC2=NC=CC(=O)N2C=C1",
            "3,4-dihydro-2H-pyrido[4,3-b][1,4]oxazine": "C1COC2=C(N1)C=NC=C2",
            "3,4-dihydro-2H-isoquinolin-1-one": "C1CNC(=O)C2=CC=CC=C21",
            "3,4-dihydro-1H-quinolin-2-one": "C1CC(=O)NC2=CC=CC=C21",
            "1,4-dihydro-3,1-benzoxazin-2-one": "C1C2=CC=CC=C2NC(=O)O1",
            "3,4-dihydro-2H-1λ6,2,4-benzothiadiazine 1,1-dioxide": "C1NC2=CC=CC=C2S(=O)(=O)N1",
            "3H-quinazolin-4-one": "C1=CC=C2C(=C1)C(=O)NC=N2",
            "4H-pyrido[3,2-b][1,4]oxazin-3-one": "C1C(=O)NC2=C(O1)C=CC=N2",
            "1,4-dihydro-1,6-naphthyridine": "C1C=CNC2=C1C=NC=C2",
            "4H-1,4-benzoxazin-3-one": "C1C(=O)NC2=CC=CC=C2O1",
            "4-hydroxyquinoline": "C1=CC=C2C(=C1)C(=O)C=CN2",
            "chromen-4-one": "C1=CC=C2C(=C1)C(=O)C=CO2",
            "5,6,7,8-tetrahydropyrido[3,4-d]pyrimidine": "C1CNCC2=NC=NC=C21",
            "8H-pyrido[2,3-d]pyrimidin-7-one": "C1=CC(=O)NC2=NC=NC=C21",
            "1H-pyrido[2,3-d]pyrimidin-2-one": "C1=CC2=C(NC(=O)N=C2)N=C1",
            "pteridine": "C1=CN=C2C(=N1)C=NC=N2",
            "3H-pteridin-4-one": "C1=CN=C2C(=N1)C(=O)NC=N2",
            "5,6,7,8-tetrahydro-3H-pteridin-4-one": "C1CNC2=C(N1)C(=O)NC=N2",
            "isoquinoline": "C1=CC=C2C=NC=CC2=C1",
            "naphthalene": "C1=CC=C2C=CC=CC2=C1",
            "quinazoline": "C1=CC=C2C(=C1)C=NC=N2",
            "phthalazine": "C1=CC=C2C=NN=CC2=C1",
            "quinoline": "C1=CC=C2C(=C1)C=CC=N2",
            "quinoxaline": "C1=CC=C2C(=C1)N=CC=N2",
            "tetrahydronaphthalene": "C1CCC2=CC=CC=C2C1",
            "1,2-dihydronaphthalene": "C1CC2=CC=CC=C2C=C1",
            "2H-phthalazin-1-one": "C1=CC=C2C(=C1)C=NNC2=O",
            "1H-1,8-naphthyridin-4-one": "C1=CC2=C(NC=CC2=O)N=C1",
            "1,3-dihydro-1,4-benzodiazepin-2-one": "C1C(=O)NC2=CC=CC=C2C=N1",
            "1,2,3,4-tetrahydrocyclopenta[b]indole": "C1CC2=C(C1)NC3=CC=CC=C23",
            "6,7,8,9-tetrahydropyrido[1,2-a]indole": "C1CCN2C(=CC3=CC=CC=C32)C1",
            "4-azatricyclo[9.4.0.03,8]pentadeca-1(15),3(8),4,6,11,13-hexaene": "C1CC2=CC=CC=C2CC3=C1C=CC=N3",
            "1,5,7,10-tetrazatricyclo[7.3.0.02,6]dodeca-2(6),3,7,9,11-pentaene": "C1=CNC2=C1N3C=CN=C3C=N2",
            "2,3,10-triazatricyclo[7.3.1.05,13]trideca-1,5(13),6,8-tetraen-4-one": "C1CNC2=CC=CC3=C2C1=NNC3=O",
            "2,3-dihydroimidazo[1,2-c]quinazoline": "C1CN2C=NC3=CC=CC=C3C2=N1",
            "3,10-diazatricyclo[6.4.1.04,13]trideca-1,4,6,8(13)-tetraen-9-one": "C1CNC(=O)C2=C3C1=CNC3=CC=C2",
            "5H-pyrido[4,3-b]indole": "C1=CC=C2C(=C1)C3=C(N2)C=CN=C3",
            "5-oxa-1,2,8-triazatricyclo[8.4.0.03,8]tetradeca-10,13-diene-9,12-dione": "C1COCC2N1C(=O)C3=CC(=O)C=CN3N2",
            "10H-phenothiazine": "C1=CC=C2C(=C1)NC3=CC=CC=C3S2",
            "6,11-dihydrobenzo[c][1]benzothiepine": "C1C2=CC=CC=C2CSC3=CC=CC=C31",
            "imidazo[2,1-b][1,3]benzothiazole": "C1=CC=C2C(=C1)N3C=CN=C3S2",
            "2,3,4,5-tetrahydropyrido[4,3-b]indol-1-one": "C1CNC(=O)C2=C1NC3=CC=CC=C32",
            "4H-imidazo[1,2-a][1,4]benzodiazepine": "C1C2=NC=CN2C3=CC=CC=C3C=N1",
            "4-oxa-1,7-diazatricyclo[7.4.0.03,7]trideca-9,12-diene-8,11-dione": "C1COC2N1C(=O)C3=CC(=O)C=CN3C2",
            "(3S)-4-oxa-1,8-diazatricyclo[8.4.0.03,8]tetradeca-10,13-diene-9,12-dione": "C1CN2[C@H](CN3C=CC(=O)C=C3C2=O)OC1",
            "2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17-hexadecahydro-1H-cyclopenta[a]phenanthrene": "C1CCC2C(C1)CCC3C2CCC4C3CCC4",
            "7,8,9,11,12,13,14,15,16,17-decahydro-6H-cyclopenta[a]phenanthrene": "C1CC2CCC3C(C2C1)CCC4=CC=CC=C34",
            "1,2,6,7,8,9,10,11,12,13,14,15,16,17-tetradecahydrocyclopenta[a]phenanthren-3-one": "C1CC2CCC3C(C2C1)CCC4=CC(=O)CCC34",
            "6,7,8,9,10,11,12,13,14,15,16,17-dodecahydrocyclopenta[a]phenanthren-3-one": "C1CC2CCC3C(C2C1)CCC4=CC(=O)C=CC34",
            "5,6-dihydrobenzo[b]carbazol-11-one": "C1C2=CC=CC=C2C(=O)C3=C1NC4=CC=CC=C43",
            "4,4a,5,5a,6,12a-hexahydrotetracene-1,11-dione": "C1C=CC(=O)C2C1CC3CC4=CC=CC=C4C(=O)C3=C2",
            "7,8,9,10-tetrahydrotetracene-5,12-dione": "C1CCC2=CC3=C(C=C2C1)C(=O)C4=CC=CC=C4C3=O",
            "5,7-dioxapentacyclo[10.8.0.02,9.04,8.013,18]icosa-14,17-dien-16-one": "C1CC2C(CC3C2OCO3)C4C1C5C=CC(=O)C=C5CC4",
            "17-oxa-3,13-diazapentacyclo[11.8.0.02,11.04,9.015,20]henicosa-1(21),2,4,6,8,10,15(20)-heptaene-14,18-dione": "C1C2=C(COC1=O)C(=O)N3CC4=CC5=CC=CC=C5N=C4C3=C2",
            "spiro[2.2]pentane": "C1CC12CC2",
            "2-azaspiro[3.3]heptane": "C1CC2(C1)CNC2",
            "1,3-diazaspiro[4.4]non-1-en-4-one": "C1CCC2(C1)C(=O)NC=N2",
            "1-oxaspiro[2.5]octane": "C1CCC2(CC1)CO2",
            "4,7-diazaspiro[2.5]octane": "C1CC12CNCCN2",
            "6-oxaspiro[4.5]decane": "C1CCC2(C1)CCCCO2",
            "1,9-diazaspiro[4.5]decan-2-one": "C1CC2(CCC(=O)N2)CNC1",
            "spiro[2,6,7,8,9,10,11,12,13,14,15,16-dodecahydro-1H-cyclopenta[a]phenanthrene-17,5'-oxolane]-2',3-dione": "C1CC2C(CCC23CCC(=O)O3)C4C1C5CCC(=O)C=C5CC4",
            "6-sulfanylidene-5,7-diazaspiro[3.4]octan-8-one": "C1CC2(C1)C(=O)NC(=S)N2",
            "1-azabicyclo[2.2.2]octane": "C1CN2CCC1CC2",
            "3,6-diazabicyclo[3.1.1]heptane": "C1C2CNCC1N2",
            "8-azabicyclo[3.2.1]octane": "C1CC2CCC(C1)N2",
            "3,8-diazabicyclo[3.2.1]octane": "C1CC2CNCC1N2",
            "adamantane": "C1C2CC3CC1CC(C2)C3",
            "tricyclo[4.2.1.03,8]nonane": "C1CC2CC3C2CC1C3",
            "1,6-diazabicyclo[3.2.1]octan-7-one": "C1CC2CN(C1)C(=O)N2",
            "1,6-diazabicyclo[3.2.1]oct-3-en-7-one": "C1C=CC2CN1C(=O)N2"
        }
    
    def has_func_group(self, mol, fg, is_SMART=True):
        """
        判断mol中是否包含官能团fg

        :param mol: str, Molecule SMILES
        :param fg: str, FG SMILES or SMART
        :param is_SMART: bool, is FG in SMART format, default True
        :return bool
        """
        if is_SMART:
            fgmol = Chem.MolFromSmarts(fg)
        else:
            fgmol = Chem.MolFromSmiles(fg)

        mol = Chem.MolFromSmiles(mol.strip())
        return len(Chem.Mol.GetSubstructMatches(mol, fgmol, uniquify=True)) > 0
    
    def get_func_group(self, mol):
        """
        检索分子mol中包含的所有官能团

        :param mol: str, 分子SMILES
        :return list(list), 所含官能团列表，包含名字,SMILES,SMARTS
        """

        try:
            fgs_in_mol_SMART = [
                [name, Chem.MolToSmiles(Chem.MolFromSmarts(fg)), fg]
                for name, fg in self.dict_fgs_SMART.items()
                if self.has_func_group(mol, fg)
            ]

            fgs_in_mol_SMILES = [
                [name, fg, None]
                for name, fg in self.dict_fgs_SMILES.items()
                if self.has_func_group(mol, fg, is_SMART=False)
            ]
            return fgs_in_mol_SMART + fgs_in_mol_SMILES
        
        except:
            return []


class AgentOCSU():

    def __init__(self, API_URL):

        self.client = OpenAI(
            api_key="EMPTY",
            base_url=API_URL,
        )
        print (f"[API Ready] {API_URL}")

        self.query_dict = {
            "SMILES": "What is the SMILES of the molecule?",
            "IUPAC": "What is the IUPAC of the molecule?",
            "Functional Groups": "Please list the functional groups of the molecule.",
            "General Caption": "Please describe this drug."
        }

        self.agent_fg = FuncGroupsAgent()
        self.visualizer = Visualizer()
    
    def chat(self, img_path, query="What is the SMILES of the molecule in the image?", temperature=0.0):

        
        chat_response = self.client.chat.completions.create(
            model="Qwen2-7B-Instruct",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": img_path
                            },
                        },
                        {"type": "text", "text": query},
                    ],
                },
            ],
            temperature=temperature
        )
        return chat_response.choices[0].message.content
    
    def extract(self, res):

        ext = dict()
        for i in res:
            if i == "SMILES":
                ext[i] = res[i][:-1].replace("The SMILES is ", "")
            elif i == "IUPAC":
                ext[i] = res[i][:-1].replace("The IUPAC is ", "")
            elif i == "Functional Groups":
                ext[i] = res[i][:-1].replace("This molecule contains several functional groups, including ", "").replace(" and", "").replace("This molecule contains the ", "").replace(" functional group", "").split(", ")
            elif i == "General Caption":
                ext[i] = res[i]
        
        return ext

    def visualize(self, smi, fgs, filename, size=[600, 600]):
        
        highlight_fg = list()
        highlight_is_smart = list()
        for fg in fgs:
            if fg in self.agent_fg.dict_fgs_SMART:
                highlight_fg.append(self.agent_fg.dict_fgs_SMART[fg])
                highlight_is_smart.append(True)
            else:
                highlight_fg.append(self.agent_fg.dict_fgs_SMILES[fg])
                highlight_is_smart.append(False)

        self.visualizer.draw_mol(smi, filename, highlight_patts=highlight_fg, is_SMART=highlight_is_smart, size=size)

    
    def analyze(self, img_path, output_path, size=[600,600]):
        
        # get result
        res = dict()
        for q_id in self.query_dict:
            print (f"[TASK] {q_id}...")
            res[q_id] = self.chat(img_path, self.query_dict[q_id])
        
        res = self.extract(res)

        # vis
        self.visualize(res["SMILES"], res["Functional Groups"], output_path, size)

        return res