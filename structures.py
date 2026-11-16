"""Turn user-supplied structures into the descriptors the prediction models use.

Aldehydes: find the aryl-CHO group, identify every substituent on that benzene ring, and sum Hammett
constants (sigma_p for para, sigma_m for meta; sigma_m is also used for sigma+ at meta, the usual approximation).
Ortho substituents have no reliable Hammett value and are reported as unsupported.
Amines: find the primary aliphatic amine and look up the Taft E_s value of its N-alkyl group.

Substituent constants: Hansch, Leo & Taft, Chem. Rev. 1991, 91, 165. Taft E_s: Taft / Kutter-Hansch scale.
"""
from rdkit import Chem, RDLogger
from rdkit.Chem import rdChemReactions

RDLogger.DisableLog('rdApp.*')  # invalid user SMILES are reported in the UI, not on stderr

# fragment SMILES (attachment point = *): (label, sigma_m, sigma_p, sigma_p_plus)
SUBSTITUENTS = {
    '*[H]': ('H', 0.00, 0.00, 0.00),
    '*C': ('CH3', -0.07, -0.17, -0.31),
    '*CC': ('Et', -0.07, -0.15, -0.30),
    '*C(C)C': ('iPr', -0.07, -0.15, -0.28),
    '*C(C)(C)C': ('tBu', -0.10, -0.20, -0.26),
    '*c1ccccc1': ('Ph', 0.06, -0.01, -0.18),
    '*OC': ('OMe', 0.12, -0.27, -0.78),
    '*OCC': ('OEt', 0.10, -0.24, -0.81),
    '*O': ('OH', 0.12, -0.37, -0.92),
    '*N': ('NH2', -0.16, -0.66, -1.30),
    '*N(C)C': ('NMe2', -0.16, -0.83, -1.70),
    '*NC(C)=O': ('NHAc', 0.21, 0.00, -0.60),
    '*SC': ('SMe', 0.15, 0.00, -0.60),
    '*F': ('F', 0.34, 0.06, -0.07),
    '*Cl': ('Cl', 0.37, 0.23, 0.11),
    '*Br': ('Br', 0.39, 0.23, 0.15),
    '*I': ('I', 0.35, 0.18, 0.14),
    '*C(F)(F)F': ('CF3', 0.43, 0.54, 0.61),
    '*C#N': ('CN', 0.56, 0.66, 0.66),
    '*[N+](=O)[O-]': ('NO2', 0.71, 0.78, 0.79),
    '*C(=O)OC': ('CO2Me', 0.37, 0.45, 0.49),
    '*C(=O)O': ('CO2H', 0.37, 0.45, 0.42),
    '*C(C)=O': ('COMe', 0.38, 0.50, 0.50),
}

# N-alkyl group (attachment point = *): (label, Taft E_s)
ALKYL_ES = {
    '*C': ('methyl', 0.00), '*CC': ('ethyl', -0.07), '*CCC': ('n-propyl', -0.36), '*CCCC': ('n-butyl', -0.39),
    '*CCCCC': ('n-pentyl', -0.40), '*C(C)C': ('isopropyl', -0.47), '*CC(C)C': ('isobutyl', -0.93),
    '*C(C)CC': ('sec-butyl', -1.13), '*C(C)(C)C': ('tert-butyl', -1.54), '*CC(C)(C)C': ('neopentyl', -1.74),
    '*C1CCCCC1': ('cyclohexyl', -0.79), '*C1CCCC1': ('cyclopentyl', -0.51), '*Cc1ccccc1': ('benzyl', -0.38),
}


def _canon(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else smiles


_SUB_LOOKUP = {_canon(k): v for k, v in SUBSTITUENTS.items()}
_ALKYL_LOOKUP = {_canon(k): v for k, v in ALKYL_ES.items()}
_ARYL_CHO = Chem.MolFromSmarts('[CX3H1](=O)[c;r6]')
_PRIMARY_AMINE = Chem.MolFromSmarts('[NX3;H2;!$(NC=[O,S,N]);!$(N-a)][CX4]')
_ANILINE = Chem.MolFromSmarts('[NX3;H2]-a')
_IMINE_RXN = rdChemReactions.ReactionFromSmarts('[CX3H1:1](=O)[c:2].[NX3H2:3][CX4:4]>>[c:2][CH1:1]=[N:3][C:4]')


def parse(smiles):
    """RDKit molecule from SMILES, or None. Accepts a single component only."""
    if not smiles or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None or len(Chem.GetMolFrags(mol)) != 1:
        return None
    return mol


def _fragment(mol, ring_atom, sub_atom):
    """Canonical SMILES of the substituent attached to ring_atom through sub_atom, with * at the attachment."""
    bond = mol.GetBondBetweenAtoms(ring_atom, sub_atom)
    pieces = Chem.FragmentOnBonds(mol, [bond.GetIdx()], dummyLabels=[(0, 0)])
    for idxs, frag in zip(Chem.GetMolFrags(pieces, sanitizeFrags=False),
                          Chem.GetMolFrags(pieces, asMols=True, sanitizeFrags=False)):
        if sub_atom in idxs:
            return Chem.MolToSmiles(frag)
    return None


def describe_aldehyde(smiles):
    """Return dict(ok, sigma_p, sigma_p_plus, label, substituents=[(position, label, sigma, sigma_plus)], messages)."""
    mol = parse(smiles)
    if mol is None:
        return {'ok': False, 'messages': ['Could not read this structure (enter a single, valid molecule).']}
    hits = mol.GetSubstructMatches(_ARYL_CHO)
    if not hits:
        return {'ok': False, 'messages': ['No aryl aldehyde (Ar–CHO on a six-membered aromatic ring) found.']}
    messages = []
    if len(hits) > 1:
        messages.append('More than one aryl-CHO group; using the first.')
    cho_c, _, ipso = hits[0]
    ring = next((r for r in mol.GetRingInfo().AtomRings() if ipso in r and len(r) == 6), None)
    if ring is None or any(mol.GetAtomWithIdx(i).GetSymbol() != 'C' for i in ring):
        return {'ok': False, 'messages': ['The aldehyde must sit on a benzene ring (heteroaromatic rings have no σ values here).']}
    pos = ring.index(ipso)
    kinds = {1: 'ortho', 2: 'meta', 3: 'para', 4: 'meta', 5: 'ortho'}
    position_of = {ring[(pos + d) % 6]: (kinds[d], d + 1) for d in kinds}  # value: (kind, ring locant with CHO at C1)
    sigma = sigma_plus = 0.0
    found, ok = [], True
    for ring_atom, (where, locant) in position_of.items():
        for nb in mol.GetAtomWithIdx(ring_atom).GetNeighbors():
            if nb.GetIdx() in ring or nb.GetIdx() == cho_c:
                continue
            if mol.GetBondBetweenAtoms(ring_atom, nb.GetIdx()).IsInRing():
                messages.append('Fused or bridged ring systems are not supported.')
                return {'ok': False, 'messages': messages}
            frag = _fragment(mol, ring_atom, nb.GetIdx())
            entry = _SUB_LOOKUP.get(frag)
            if where == 'ortho':
                ok = False
                messages.append(f'Ortho substituent ({entry[0] if entry else frag}) has no Hammett value; enter σ manually.')
                found.append((locant, entry[0] if entry else frag, None, None))
            elif entry is None:
                ok = False
                messages.append(f'{where} substituent {frag} is not in the σ table; enter σ manually.')
                found.append((locant, frag, None, None))
            else:
                s, sp = (entry[2], entry[3]) if where == 'para' else (entry[1], entry[1])
                sigma += s
                sigma_plus += sp
                found.append((locant, entry[0], s, sp))
    # Number the ring in whichever direction gives the lowest locants (CHO carbon's ring atom = C1).
    mirror = {2: 6, 3: 5, 4: 4, 5: 3, 6: 2}
    if sorted(mirror[f[0]] for f in found) < sorted(f[0] for f in found):
        found = [(mirror[f[0]], *f[1:]) for f in found]
    found.sort()
    label = ', '.join(f'{locant}-{lab}' for locant, lab, *_ in found if lab != 'H') or 'H'
    if not found:
        found.append((4, 'H', 0.0, 0.0))
    if len([f for f in found if f[1] != 'H']) > 1:
        messages.append('Several substituents: σ values are summed (Hammett additivity).')
    messages = list(dict.fromkeys(messages))
    return {'ok': ok, 'scaffold': True, 'sigma_p': sigma, 'sigma_p_plus': sigma_plus, 'label': label, 'substituents': found, 'messages': messages}


def describe_amine(smiles):
    """Return dict(ok, taft_Es, label, messages)."""
    mol = parse(smiles)
    if mol is None:
        return {'ok': False, 'messages': ['Could not read this structure (enter a single, valid molecule).']}
    if mol.HasSubstructMatch(_ANILINE) and not mol.HasSubstructMatch(_PRIMARY_AMINE):
        return {'ok': False, 'messages': ['Aromatic amines (anilines) are far weaker nucleophiles than the alkylamines '
                                          'in the training data; they cannot be predicted.']}
    hits = mol.GetSubstructMatches(_PRIMARY_AMINE)
    if not hits:
        return {'ok': False, 'messages': ['No primary aliphatic amine (R–NH₂ on an sp³ carbon) found.']}
    messages = ['More than one primary amine; using the first.'] if len(hits) > 1 else []
    n_idx, c_idx = hits[0]
    frag = _fragment(mol, n_idx, c_idx)
    entry = _ALKYL_LOOKUP.get(frag)
    if entry is None:
        return {'ok': False, 'scaffold': True, 'label': frag, 'messages': messages + [f'N-alkyl group {frag} is not in the Taft Eₛ table; enter Eₛ manually.']}
    return {'ok': True, 'scaffold': True, 'taft_Es': entry[1], 'label': entry[0], 'messages': messages}


def imine_product(aldehyde_smiles, amine_smiles):
    """SMILES of the imine from the first aryl-CHO and primary amine, or None."""
    ald, amine = parse(aldehyde_smiles), parse(amine_smiles)
    if ald is None or amine is None:
        return None
    products = _IMINE_RXN.RunReactants((ald, amine))
    for (product,) in products:
        try:
            Chem.SanitizeMol(product)
            return Chem.MolToSmiles(product)
        except Exception:
            continue
    return None
