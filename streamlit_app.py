import base64
import re
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats as scipy_stats
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D

try:
    from streamlit_ketcher import st_ketcher
    HAS_KETCHER = True
except ImportError:  # drawing is optional; SMILES and name entry still work
    HAS_KETCHER = False

from ml import (DESCRIPTORS, add_descriptors, cross_validate, predict_conversion_gbm, predict_conversion_gp,
                predict_ln_k, train_direct_gbm, train_kinetic_gp)
from structures import describe_aldehyde, describe_amine, imine_product, parse as parse_smiles
from analysis import R_GAS, activation_parameters, conversion, fit_kinetics, linearised, model_comparison, ols, time_to_reach

CSV_PATH = Path(__file__).with_name('imine_condensation_dataset_for_students.csv')
# Para-substituents ordered by Hammett sigma_p (electron-donating -> electron-withdrawing).
R_GROUP_ORDER = ['NH2', 'OMe', 'CH3', 'H', 'Cl', 'CF3', 'NO2']
HAMMETT_SIGMA_P = {'NH2': -0.66, 'OMe': -0.27, 'CH3': -0.17, 'H': 0.00, 'Cl': 0.23, 'CF3': 0.54, 'NO2': 0.78}
# Brown-Okamoto sigma_p+ (Hansch, Leo & Taft, Chem. Rev. 1991, 91, 165).
HAMMETT_SIGMA_P_PLUS = {'NH2': -1.30, 'OMe': -0.78, 'CH3': -0.31, 'H': 0.00, 'Cl': 0.11, 'CF3': 0.61, 'NO2': 0.79}
# Taft steric constants E_s of the alkyl group on nitrogen.
TAFT_ES = {'n-butylamine': -0.39, 'sec-butylamine': -1.13, 'tert-butylamine': -1.54}
SUB_COLORS = {'NH2': '#3D7BE0', 'OMe': '#5F9BEF', 'CH3': '#8DB7F2', 'H': '#B9B9C3', 'Cl': '#F0A58F', 'CF3': '#E8735F', 'NO2': '#D2433A'}
AMINE_COLORS = {'n-butylamine': '#7CC79A', 'sec-butylamine': '#B48CE0', 'tert-butylamine': '#E0A25A'}
TEMP_COLORS = ['#5B9BF0', '#E8B04B', '#E8615C']
# Heatmap scale: blue (low) → light neutral → red (high), matching the app's cool/warm blue and red.
BLUE_RED_SCALE = ['#1F4E9A', '#3D7BE0', '#8DB7F2', '#E6E3E1', '#F0A58F', '#E8615C', '#B3282D']
CONVERSION_SCALE = BLUE_RED_SCALE
RATE_SCALE = BLUE_RED_SCALE
AMINE_NOTES = {'n-butylamine': 'primary carbon on N', 'sec-butylamine': 'secondary carbon on N',
               'tert-butylamine': 'tertiary carbon on N'}
SMILES_COLUMNS = {'aldehyde_SMILES', 'amine_SMILES', 'imine_SMILES'}
# Generic structures: atom-map 1 is drawn as "Ar", atom-map 2 as "R'".
GENERIC = {'aldehyde': '[*:1]C=O', 'amine': 'N[*:2]', 'hemiaminal': '[*:1]C(O)N[*:2]',
           'imine': '[*:1]C=N[*:2]', 'water': 'O'}
REQUIRED = {
    'aldehyde_name', 'aldehyde_R_group', 'amine_name', 'solvent',
    'temperature_C', 'time_min', 'conversion_percent',
}

st.set_page_config(page_title='Imine Reaction Explorer', layout='wide')

APP_CSS = '''<style>
/* Layout */
.block-container {padding-top: 2.4rem; padding-bottom: 2.5rem; max-width: 1440px;}
/* Page header */
.app-header {padding: 0.2rem 0 1.1rem; margin-bottom: 0.4rem; border-bottom: 1px solid rgba(127,127,127,0.2);}
.app-header .overline {font-size: 0.72rem; font-weight: 600; letter-spacing: 0.14em; text-transform: uppercase; color: #7AB8F5;}
.app-header h1 {font-size: 2.15rem; font-weight: 700; letter-spacing: -0.02em; line-height: 1.15; margin: 0.25rem 0 0.4rem; padding: 0;}
.app-header p {margin: 0; font-size: 0.95rem; opacity: 0.72; max-width: 62rem;}
.app-header .chips {display: flex; flex-wrap: wrap; gap: 0.4rem; margin-top: 0.85rem;}
.app-header .chip {font-size: 0.78rem; padding: 0.2rem 0.65rem; border-radius: 999px;
                   border: 1px solid rgba(127,127,127,0.28); background: rgba(127,127,127,0.08); white-space: nowrap;}
.app-header .chip b {font-weight: 600; color: #8EC5FA;}
/* Metric cards */
[data-testid="stMetric"] {background: rgba(127,127,127,0.07); border: 1px solid rgba(127,127,127,0.18);
                          border-radius: 0.75rem; padding: 0.75rem 1rem 0.7rem;}
[data-testid="stMetricLabel"] p {font-size: 0.74rem; font-weight: 500; letter-spacing: 0.06em; text-transform: uppercase; opacity: 0.72;}
[data-testid="stMetricValue"] {letter-spacing: -0.01em;}
/* Tabs */
.stTabs [role="tablist"] {gap: 0.1rem; flex-wrap: wrap; row-gap: 0.3rem; padding-bottom: 0.5rem; border-bottom: 1px solid rgba(127,127,127,0.22); box-shadow: none;}
.stTabs [data-testid="stTab"] {padding: 0.38rem 0.6rem; border-radius: 0.6rem; border: 1px solid transparent; transition: background 0.15s;}
.stTabs [data-testid="stTab"] p {font-size: 0.86rem; font-weight: 500; white-space: nowrap;}
.stTabs [data-testid="stTab"]:hover {background: rgba(127,127,127,0.10);}
.stTabs [data-testid="stTab"][aria-selected="true"] {background: rgba(122,184,245,0.15); border-color: rgba(122,184,245,0.5);}
.stTabs .react-aria-SelectionIndicator {display: none;}
.stTabs [role="tabpanel"] {padding-top: 1.1rem;}
/* Headings and captions */
.stMainBlockContainer h3 {letter-spacing: -0.01em; padding-top: 0.9rem;}
[data-testid="stCaptionContainer"] {line-height: 1.5;}
/* Sidebar */
section[data-testid="stSidebar"] h2 {font-size: 1.15rem; letter-spacing: -0.01em;}
section[data-testid="stSidebar"] h3 {font-size: 0.74rem; font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase; opacity: 0.8;}
section[data-testid="stSidebar"] [data-testid="stExpander"] summary p {font-weight: 500;}
/* Expanders, tables, charts */
[data-testid="stExpander"] details {border-radius: 0.65rem;}
[data-testid="stExpander"] summary p {font-weight: 500;}
[data-testid="stPlotlyChart"] {border: 1px solid rgba(127,127,127,0.14); border-radius: 0.75rem; padding: 0.35rem 0.4rem 0.1rem;}
/* Small screens: clear Streamlit's top bar */
@media (max-width: 640px) {.block-container {padding-top: 3.6rem;} .app-header h1 {font-size: 1.75rem;}}
/* Footer */
.app-footer {margin-top: 2.5rem; padding-top: 1rem; border-top: 1px solid rgba(127,127,127,0.2);
             font-size: 0.78rem; opacity: 0.6; display: flex; flex-wrap: wrap; gap: 0.3rem 1.25rem;}
</style>'''
st.html(APP_CSS)
header_box = st.container()

@st.cache_data
def read_csv(source):
    return pd.read_csv(source)


@st.cache_data
def structure_svg(smiles, bond_length=26):
    """Inline RDKit SVG of `smiles`. The carbon skeleton uses currentColor, so it follows the page theme."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return f'<code>{escape(smiles)}</code>'
    for atom in mol.GetAtoms():
        label = {1: 'Ar', 2: "R'"}.get(atom.GetAtomMapNum())
        if label:
            atom.SetAtomMapNum(0)
            atom.SetProp('atomLabel', label)
    drawer = rdMolDraw2D.MolDraw2DSVG(-1, -1)  # canvas sized to the molecule, so every structure shares one scale
    options = drawer.drawOptions()
    options.clearBackground = False
    options.fixedBondLength = bond_length
    options.bondLineWidth = 2
    options.minFontSize = 14
    options.padding = 0.12
    # Heteroatom colours chosen to read on both light and dark backgrounds.
    options.updateAtomPalette({0: (0, 0, 0), 7: (0.35, 0.58, 0.98), 8: (0.93, 0.36, 0.34), 9: (0.30, 0.74, 0.45), 17: (0.30, 0.74, 0.45),
                                16: (0.90, 0.76, 0.29), 35: (0.85, 0.53, 0.31), 53: (0.69, 0.49, 0.86)})
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return svg[svg.index('<svg'):].replace('#000000', 'currentColor').replace('\n', ' ')


def parse_structure(smiles):
    return parse_smiles(smiles)


def structure_figure(smiles, caption):
    return SCHEME_CSS + f'<div class="scheme"><figure>{structure_svg(smiles, 24)}<figcaption>{escape(caption)}</figcaption></figure></div>'


@st.cache_data(show_spinner='Looking up the name on PubChem…', ttl=86400)
def pubchem_smiles(name):
    """SMILES for a compound name from PubChem, or an error message."""
    import json
    import urllib.parse
    import urllib.request
    url = ('https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/' + urllib.parse.quote(name.strip())
           + '/property/IsomericSMILES,CanonicalSMILES/JSON')
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            props = json.load(response)['PropertyTable']['Properties'][0]
    except Exception as error:
        return None, f'PubChem lookup failed ({error.__class__.__name__}). Check the name or type SMILES instead.'
    for key in ('IsomericSMILES', 'SMILES', 'CanonicalSMILES', 'ConnectivitySMILES'):
        if props.get(key):
            return props[key], None
    return None, 'PubChem returned no SMILES for that name.'


def structure_input(role, default_smiles, default_name):
    """Let the user give a structure by SMILES, drawing or name. Returns a SMILES string (may be empty)."""
    modes = ['Type SMILES', 'Draw', 'Look up name'] if HAS_KETCHER else ['Type SMILES', 'Look up name']
    mode = st.radio(f'Enter the {role} by', modes, horizontal=True, key=f'mode_{role}')
    if mode == 'Type SMILES':
        return st.text_input('SMILES', default_smiles, key=f'smiles_{role}', help='e.g. O=Cc1ccc(Br)cc1 or NCC(C)C')
    if mode == 'Draw':
        st.caption('Draw the structure, then press **Apply** under the editor.')
        return st_ketcher(default_smiles, height=380, key=f'ketcher_{role}') or ''
    name = st.text_input('Compound name', default_name, key=f'name_{role}')
    if not name.strip():
        return ''
    smiles, error = pubchem_smiles(name)
    if error:
        st.warning(error)
        return ''
    st.caption(f'PubChem: {smiles}')
    return smiles


SCHEME_CSS = '''<style>
.scheme {display:flex; flex-wrap:wrap; align-items:center; justify-content:center; gap:0.4rem 1rem; margin:0.5rem 0 1rem;}
.scheme figure {margin:0; display:flex; flex-direction:column; align-items:center;}
.scheme figcaption {font-size:0.82rem; opacity:0.8; text-align:center; max-width:14rem;}
.scheme .op {font-size:1.8rem; opacity:0.85;}
.scheme .arrow {display:flex; flex-direction:column; align-items:center; min-width:7rem; font-size:0.8rem; opacity:0.9;}
.scheme .arrow b {font-size:2.4rem; line-height:1; font-weight:400;}
.substrates {overflow-x:auto; margin-bottom:1.5rem;}
.gallery {display:grid; grid-template-columns:repeat(7, minmax(6.5rem, 1fr)); gap:0.5rem 0.5rem;}
.gallery figure {margin:0; padding:0.5rem 0.25rem 0.4rem; border-radius:0.5rem; background:rgba(127,127,127,0.1);
                 display:flex; flex-direction:column; align-items:center; justify-content:flex-end;}
.gallery svg {width:100%; height:auto; display:block;}
.gallery figcaption {font-size:0.8rem; line-height:1.3; text-align:center; overflow-wrap:anywhere;}
.gallery small {opacity:0.75;}
.trend {margin:0.2rem 0.25rem 0;}
.trend .bar {position:relative; height:6px; margin-right:14px; border-radius:3px 0 0 3px;
             background:linear-gradient(90deg, var(--from), var(--to));}
.trend .bar::after {content:''; position:absolute; right:-14px; top:-6px; border-left:14px solid var(--to);
                    border-top:9px solid transparent; border-bottom:9px solid transparent;}
.trend .labels {display:flex; justify-content:space-between; gap:1rem; font-size:0.8rem; margin-top:0.35rem;}
.trend .labels span:nth-child(2) {opacity:0.75; text-align:center;}
.trend .labels span:first-child, .trend .labels span:last-child {white-space:nowrap;}
</style>'''


def scheme(parts, bond_length=26):
    """parts: ('mol', smiles, caption) | ('op', text) | ('arrow', above, below)."""
    html = []
    for part in parts:
        if part[0] == 'mol':
            html.append(f'<figure>{structure_svg(part[1], bond_length)}<figcaption>{escape(part[2])}</figcaption></figure>')
        elif part[0] == 'op':
            html.append(f'<span class="op">{escape(part[1])}</span>')
        else:
            html.append(f'<div class="arrow"><span>{escape(part[1])}</span><b>&#8652;</b><span>{escape(part[2])}</span></div>')
    st.markdown(SCHEME_CSS + '<div class="scheme">' + ''.join(html) + '</div>', unsafe_allow_html=True)


def gallery(items, trend):
    """items: (smiles, title, subtitle, tooltip); trend: (left label, middle label, right label, from colour, to colour)."""
    svgs = [structure_svg(item[0], 20) for item in items]
    widths = [float(m.group(1)) if (m := re.search(r"width='([\d.]+)px'", svg)) else 100.0 for svg in svgs]
    widest = max(widths)
    # Same scale for every card in the row: each drawing gets the share of card width its natural size implies.
    cards = ''.join(f'<figure title="{escape(tip)}"><div style="width:{96 * w / widest:.0f}%; max-width:{w:.0f}px">{svg}</div>'
                    f'<figcaption><b>{escape(title)}</b><br><small>{escape(sub)}</small></figcaption></figure>'
                    for svg, w, (_, title, sub, tip) in zip(svgs, widths, items))
    left, middle, right, start, end = trend
    # Both substrate rows share a 7-column grid, so cards are the same size; the arrow spans the occupied columns.
    arrow = (f'<div class="trend" style="--from:{start}; --to:{end}; grid-column:1 / span {len(items)}"><div class="bar"></div><div class="labels">'
             f'<span>{escape(left)}</span><span>{escape(middle)}</span><span>{escape(right)}</span></div></div>')
    st.markdown(SCHEME_CSS + f'<div class="substrates"><div class="gallery">{cards}{arrow}</div></div>', unsafe_allow_html=True)

filters_box, settings_box, source_box = st.sidebar.container(), st.sidebar.container(), st.sidebar.container()
with source_box.expander('Data source'):
    own_csv = st.file_uploader('Analyse your own CSV', type='csv',
                               help='Needs the same columns as the workshop file. SMILES columns are optional.')
    st.caption('Without an upload the bundled workshop dataset is used.')
if own_csv is not None:
    data = read_csv(own_csv)
elif CSV_PATH.exists():
    data = read_csv(CSV_PATH)
else:
    st.info('Upload a CSV in the sidebar (Data source) to begin.')
    st.stop()

missing = REQUIRED - set(data.columns)
if missing:
    st.error(f'Missing required columns: {", ".join(sorted(missing))}')
    st.stop()

data = data.copy()
for column in ('temperature_C', 'time_min', 'conversion_percent'):
    data[column] = pd.to_numeric(data[column], errors='coerce')
data = data.dropna(subset=list(REQUIRED))
chips = ''.join(f'<span class="chip"><b>{n}</b> {label}</span>' for n, label in (
    (len(data), 'measurements'), (data.aldehyde_name.nunique(), 'aldehydes'), (data.amine_name.nunique(), 'amines'),
    (data.solvent.nunique(), 'solvents'), (data.temperature_C.nunique(), 'temperatures'), (data.time_min.nunique(), 'time points')))
header_box.html(f'''<div class="app-header">
  <div class="overline">Physical organic chemistry · Imine condensation</div>
  <h1>Imine Reaction Explorer</h1>
  <p>{len(data)} imine-condensation measurements: {data.aldehyde_name.nunique()} aldehydes × {data.amine_name.nunique()} amines ×
     {data.solvent.nunique()} solvents × {data.temperature_C.nunique()} temperatures × {data.time_min.nunique()} times.</p>
  <div class="chips">{chips}</div>
</div>''')

SUBSCRIPTS = str.maketrans('0123456789', '₀₁₂₃₄₅₆₇₈₉')
r_options = sorted(data.aldehyde_R_group.unique(), key=lambda r: (HAMMETT_SIGMA_P.get(r, 99), r))
amine_options = sorted(data.amine_name.unique(), key=lambda n: (list(AMINE_NOTES).index(n) if n in AMINE_NOTES else 99, n))
FILTERS = {  # session key: (label, options, format, help)
    'f_r': ('Aldehyde substituent (para-R)', r_options, lambda r: r.translate(SUBSCRIPTS),
            'Ordered from electron-donating (left) to electron-withdrawing (right).'),
    'f_amine': ('Amine', amine_options, str, 'Ordered by increasing steric bulk next to nitrogen.'),
    'f_solvent': ('Solvent', sorted(data.solvent.unique()), str, None),
    'f_temp': ('Temperature', sorted(data.temperature_C.unique()), lambda t: f'{t:g} °C', None),
    'f_time': ('Reaction time', sorted(data.time_min.unique()), lambda t: f'{t:g} min', None),
}


def reset_filters():
    for key, (_, options, _, _) in FILTERS.items():
        st.session_state[key] = list(options)


def fair_comparison():
    """Keep all reactants but fix one solvent, the middle temperature and the longest time."""
    reset_filters()
    for key, pick in (('f_solvent', 0), ('f_temp', len(FILTERS['f_temp'][1]) // 2), ('f_time', -1)):
        st.session_state[key] = [FILTERS[key][1][pick]]


for key, (_, options, _, _) in FILTERS.items():
    st.session_state.setdefault(key, list(options))

with filters_box:
    st.header('Filters')
    st.caption('Click a button to include or exclude it. Highlighted = included. Every tab uses these filters.')
    b1, b2 = st.columns(2)
    b1.button('Show all', on_click=reset_filters, width='stretch')
    b2.button('Fair test', on_click=fair_comparison, width='stretch',
              help='One solvent, one temperature and one time, so reactants are compared under identical conditions.')
    selections = {}
    for group, keys in (('Reactants', ('f_r', 'f_amine')), ('Conditions', ('f_solvent', 'f_temp', 'f_time'))):
        st.subheader(group, divider='gray')
        for key in keys:
            label, options, fmt, help_text = FILTERS[key]
            selections[key] = st.pills(label, options, selection_mode='multi', format_func=fmt, key=key, help=help_text)
            if not selections[key]:
                st.caption(':orange[Nothing selected, so no data. Pick at least one.]')

MODELS = {'Best fit (global order n)': None, 'First order': 1.0, 'Second order (equimolar)': 2.0}
with settings_box.expander('Analysis settings'):
    model_name = st.radio('Kinetic model', list(MODELS),
                          help='dx/dt = k_app (1 − x)^n. "Best fit" finds one order n shared by every time course.')
    sigma_name = st.radio('Hammett constant', ['σp', 'σp⁺'], horizontal=True,
                          help='σp⁺ (Brown–Okamoto) applies when a positive charge builds up in conjugation with the ring.')
    c0 = st.number_input('Initial concentration c₀ (M)', min_value=0.0, value=0.0, step=0.05, format='%.3f',
                         help='Optional, equimolar reagents. For n ≠ 1 converts k_app into k = k_app / c₀ⁿ⁻¹. Leave 0 if unknown.')
    export_format = st.radio('Chart download format', ['svg', 'png'], horizontal=True,
                             help='Used by the camera button above each chart.')
SIGMA = HAMMETT_SIGMA_P if sigma_name == 'σp' else HAMMETT_SIGMA_P_PLUS


@st.cache_data(show_spinner='Fitting kinetic models…')
def kinetic_models(frame):
    return model_comparison(frame)


@st.cache_data(show_spinner=False)
def kinetic_fit(frame, order):
    return fit_kinetics(frame, order)


model_table, order_profile_df, order_ci = kinetic_models(data)
best_order = float(model_table.loc[model_table.Model.str.startswith('Global'), 'Order n'].iloc[0])
ORDER = MODELS[model_name] if MODELS[model_name] is not None else best_order
kin_all = kinetic_fit(data, ORDER)
if abs(ORDER - 1) < 1e-9 or c0 <= 0:
    kin_all['k_per_s'] = kin_all.k_app_per_min / 60
    K_UNIT = 's⁻¹'
else:
    kin_all['k_per_s'] = kin_all.k_app_per_min / 60 / c0 ** (ORDER - 1)
    K_UNIT = f'M^{1 - ORDER:.2f} s⁻¹'
act_all = activation_parameters(kin_all)


@st.cache_resource(show_spinner='Training prediction models…')
def prediction_models(frame, order):
    kin_desc = add_descriptors(fit_kinetics(frame, order), HAMMETT_SIGMA_P, HAMMETT_SIGMA_P_PLUS, TAFT_ES).dropna(subset=DESCRIPTORS)
    rows = add_descriptors(frame, HAMMETT_SIGMA_P, HAMMETT_SIGMA_P_PLUS, TAFT_ES).dropna(subset=DESCRIPTORS)
    return train_kinetic_gp(kin_desc), train_direct_gbm(rows), kin_desc


@st.cache_data(show_spinner='Cross-validating…')
def prediction_cv(frame, order):
    return cross_validate(frame, order, HAMMETT_SIGMA_P, HAMMETT_SIGMA_P_PLUS, TAFT_ES)


def show(fig, name, height=None):
    """Render a Plotly chart with a consistent look and a publication-quality download button."""
    fig.update_layout(margin=dict(l=10, r=10, b=10, t=max(50, fig.layout.margin.t or 0)), legend_title_text='',
                      font=dict(family='Inter, sans-serif', size=13), legend=dict(font_size=12, bgcolor='rgba(0,0,0,0)'),
                      hoverlabel=dict(font_family='Inter, sans-serif', font_size=12),
                      **({'height': height} if height else {}))
    axis_style = dict(gridcolor='rgba(127,127,127,0.16)', zeroline=False, showline=True, linecolor='rgba(127,127,127,0.4)',
                      ticks='outside', tickcolor='rgba(127,127,127,0.4)', ticklen=4, title_font_size=13)
    fig.update_xaxes(**axis_style)
    fig.update_yaxes(**axis_style)
    st.plotly_chart(fig, width='stretch', config={'displaylogo': False, 'toImageButtonOptions': {
        'format': export_format, 'filename': name, 'scale': 3}})


def sub_label(r):
    return r.translate(SUBSCRIPTS)


aldehydes = sorted(data[data.aldehyde_R_group.isin(selections['f_r'])].aldehyde_name.unique())
amines = [a for a in amine_options if a in selections['f_amine']]
solvents = selections['f_solvent']
temperatures = sorted(selections['f_temp'])
times = sorted(selections['f_time'])

filtered = data[
    data.aldehyde_name.isin(aldehydes)
    & data.amine_name.isin(amines)
    & data.solvent.isin(solvents)
    & data.temperature_C.isin(temperatures)
    & data.time_min.isin(times)
].copy()
if filtered.empty:
    st.warning('No measurements match these filters. Select at least one button in each sidebar group, or click Show all.')
    st.stop()

st.caption(f'{len(filtered)} measurements shown out of {len(data)}. Summaries are descriptive; they are not predictions for unmeasured conditions.')
k1, k2, k3, k4 = st.columns(4)
k1.metric('Mean conversion', f'{filtered.conversion_percent.mean():.1f}%')
k2.metric('Median conversion', f'{filtered.conversion_percent.median():.1f}%')
k3.metric('Highest conversion', f'{filtered.conversion_percent.max():.1f}%')
k4.metric('Experiments shown', f'{len(filtered)}')

tab0, tab_p, tab1, tab_k, tab_h, tab_e, tab2, tab3, tab4 = st.tabs(['Reaction scheme', 'Predict', 'Time courses', 'Kinetics', 'Hammett & Taft',
                                                         'Activation parameters', 'Reactivity map', 'Matched comparisons', 'Data & guide'])

kin = kin_all[kin_all.aldehyde_name.isin(aldehydes) & kin_all.amine_name.isin(amines)
              & kin_all.solvent.isin(solvents) & kin_all.temperature_C.isin(temperatures)].copy()
act = act_all[act_all.aldehyde_name.isin(aldehydes) & act_all.amine_name.isin(amines) & act_all.solvent.isin(solvents)].copy()
ald_by_sigma = sorted(aldehydes, key=lambda a: HAMMETT_SIGMA_P.get(data.loc[data.aldehyde_name == a, 'aldehyde_R_group'].iloc[0], 99))
R_OF = data.drop_duplicates('aldehyde_name').set_index('aldehyde_name').aldehyde_R_group.to_dict()

with tab0:
    st.subheader('General reaction')
    scheme([('mol', GENERIC['aldehyde'], 'aryl aldehyde'), ('op', '+'), ('mol', GENERIC['amine'], 'primary amine'),
            ('arrow', 'addition', ''), ('mol', GENERIC['hemiaminal'], 'hemiaminal (intermediate, not measured)'),
            ('arrow', 'dehydration', ''), ('mol', GENERIC['imine'], 'imine (Schiff base)'), ('op', '+'),
            ('mol', GENERIC['water'], 'water')], bond_length=36)
    st.caption('The amine adds to the carbonyl carbon to give a hemiaminal, which loses water to form the C=N bond. '
               'Both steps are reversible. Electron-withdrawing para-substituents make the carbonyl carbon more electrophilic; '
               'bulky groups next to the amine nitrogen hinder the addition.')

    if not SMILES_COLUMNS.issubset(data.columns):
        st.info('Add aldehyde_SMILES, amine_SMILES and imine_SMILES columns to the CSV to draw specific reactions.')
    else:
        st.subheader('A specific reaction from the dataset')
        c1, c2, c3, c4 = st.columns(4)
        ald_options = [a for a in data.drop_duplicates('aldehyde_name').assign(
            sigma=lambda d: d.aldehyde_R_group.map(HAMMETT_SIGMA_P)).sort_values('sigma').aldehyde_name if a in aldehydes]
        pick_ald = c1.selectbox('Aldehyde', ald_options)
        pick_amine = c2.selectbox('Amine', amines)
        pick_solvent = c3.selectbox('Solvent', solvents)
        pick_temp = c4.selectbox('Temperature (°C)', temperatures, index=len(temperatures) // 2,
                                 format_func=lambda t: f'{t:g}')
        rxn = data[(data.aldehyde_name == pick_ald) & (data.amine_name == pick_amine)]
        row = rxn.iloc[0]
        scheme([('mol', row.aldehyde_SMILES, pick_ald), ('op', '+'), ('mol', row.amine_SMILES, pick_amine),
                ('arrow', pick_solvent, f'{pick_temp:g} °C'), ('mol', row.imine_SMILES, 'imine product'),
                ('op', '+'), ('mol', 'O', 'water')])
        course = rxn[(rxn.solvent == pick_solvent) & (rxn.temperature_C == pick_temp)].sort_values('time_min')
        if course.empty:
            st.info('This combination was not measured.')
        else:
            st.markdown('**Measured conversion**')
            for column, (_, point) in zip(st.columns(len(course)), course.iterrows()):
                column.metric(f'{point.time_min:g} min', f'{point.conversion_percent:.1f}%')

        st.subheader('Substrates')
        st.markdown('**Aldehydes**: para-substituted benzaldehydes, 4-R-C₆H₄-CHO (hover a card for the name)')
        alds = data.drop_duplicates('aldehyde_name')
        alds = alds.assign(order=alds.aldehyde_R_group.map({r: i for i, r in enumerate(R_GROUP_ORDER)})).sort_values('order')
        gallery([(r.aldehyde_SMILES, f'R = {r.aldehyde_R_group.translate(SUBSCRIPTS)}',
                  f'σp {HAMMETT_SIGMA_P[r.aldehyde_R_group]:+.2f}'.replace('-', '−')
                  if r.aldehyde_R_group in HAMMETT_SIGMA_P else '', r.aldehyde_name)
                 for r in alds.itertuples()],
                ('electron-donating', 'more electrophilic C=O →', 'electron-withdrawing',
                 '#5B9BF0', '#E8615C'))
        st.markdown('**Amines** (butylamine isomers)')
        amine_rows = data.drop_duplicates('amine_name').set_index('amine_name')
        gallery([(amine_rows.loc[a, 'amine_SMILES'], a, AMINE_NOTES.get(a, ''), a)
                 for a in sorted(amine_rows.index, key=lambda n: (list(AMINE_NOTES).index(n) if n in AMINE_NOTES else 99, n))],
                ('less hindered', 'bulkier next to N →', 'more hindered',
                 '#7CC79A', '#E0A25A'))

        with st.expander('Substituent constants and molecular descriptors'):
            rows = []
            for r in alds.itertuples():
                mol = Chem.MolFromSmiles(r.aldehyde_SMILES)
                rows.append({'Compound': r.aldehyde_name, 'Role': 'aldehyde', 'R': sub_label(r.aldehyde_R_group),
                             'σp': HAMMETT_SIGMA_P.get(r.aldehyde_R_group), 'σp⁺': HAMMETT_SIGMA_P_PLUS.get(r.aldehyde_R_group),
                             'Taft Eₛ': None, 'Formula': rdMolDescriptors.CalcMolFormula(mol), 'MW': Descriptors.MolWt(mol),
                             'cLogP': Crippen.MolLogP(mol), 'TPSA': rdMolDescriptors.CalcTPSA(mol), 'SMILES': r.aldehyde_SMILES})
            for name, row in amine_rows.iterrows():
                mol = Chem.MolFromSmiles(row.amine_SMILES)
                rows.append({'Compound': name, 'Role': 'amine', 'R': '', 'σp': None, 'σp⁺': None, 'Taft Eₛ': TAFT_ES.get(name),
                             'Formula': rdMolDescriptors.CalcMolFormula(mol), 'MW': Descriptors.MolWt(mol),
                             'cLogP': Crippen.MolLogP(mol), 'TPSA': rdMolDescriptors.CalcTPSA(mol), 'SMILES': row.amine_SMILES})
            st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch', column_config={
                'σp': st.column_config.NumberColumn(format='%+.2f'), 'σp⁺': st.column_config.NumberColumn(format='%+.2f'),
                'Taft Eₛ': st.column_config.NumberColumn(format='%+.2f'), 'MW': st.column_config.NumberColumn('MW (g/mol)', format='%.2f'),
                'cLogP': st.column_config.NumberColumn('cLogP (Crippen)', format='%.2f'),
                'TPSA': st.column_config.NumberColumn('TPSA (Å²)', format='%.1f')})
            st.caption('σ values: Hansch, Leo & Taft, Chem. Rev. 1991, 91, 165. Eₛ: Taft steric constants of the N-alkyl group. '
                       'MW, cLogP and TPSA calculated with RDKit.')

with tab_p:
    st.subheader('Predict conversion')
    st.caption('Choose reactants and conditions; machine-learning models trained on this dataset predict conversion, rate and time to target. '
               'Reactants are described by σp, σp⁺ and Taft Eₛ, so you can also try a substituent or amine that was not measured.')
    gp_model, gbm_model, kin_desc = prediction_models(data, ORDER)
    T_LO, T_HI = float(data.temperature_C.min()), float(data.temperature_C.max())
    TIME_HI = float(data.time_min.max())
    CUSTOM = 'Custom structure…'
    known_ald_smiles = {Chem.CanonSmiles(v): k for k, v in data.drop_duplicates('aldehyde_name')
                        .set_index('aldehyde_name').aldehyde_SMILES.items()} if SMILES_COLUMNS.issubset(data.columns) else {}
    known_amine_smiles = {Chem.CanonSmiles(v): k for k, v in data.drop_duplicates('amine_name')
                          .set_index('amine_name').amine_SMILES.items()} if SMILES_COLUMNS.issubset(data.columns) else {}

    i1, i2 = st.columns(2)
    with i1:
        ald_known = sorted(R_OF.items(), key=lambda kv: HAMMETT_SIGMA_P.get(kv[1], 99))
        ald_pick = st.selectbox('Aldehyde', [a for a, _ in ald_known] + [CUSTOM], index=3 if len(ald_known) > 3 else 0, key='p_ald')
        if ald_pick == CUSTOM:
            ald_smiles = structure_input('aldehyde', 'O=Cc1ccc(Br)cc1', '4-bromobenzaldehyde')
            info = describe_aldehyde(ald_smiles) if ald_smiles else {'ok': False, 'messages': ['Enter a structure.']}
            if ald_smiles and parse_structure(ald_smiles) is not None:
                st.markdown(structure_figure(ald_smiles, info.get('label') or ''), unsafe_allow_html=True)
            for message in info['messages']:
                (st.caption if info['ok'] else st.warning)(message)
            if info.get('substituents'):
                st.caption(' · '.join(f'{pos}-{lab}: σ {sig:+.2f} / σ⁺ {sigp:+.2f}' if sig is not None else f'{pos}-{lab}: no value'
                                      for pos, lab, sig, sigp in info['substituents']).replace('-0', '−0'))
            c1, c2 = st.columns(2)
            stamp = abs(hash(ald_smiles)) % 10 ** 8
            p_sp = c1.number_input('Σσ (σp / σm)', -2.0, 2.0, float(info.get('sigma_p', 0.0)), 0.01, key=f'p_sp_{stamp}',
                                   help='Filled in from the structure when every substituent is recognised; edit to override.')
            p_spp = c2.number_input('Σσ⁺', -3.0, 2.0, float(info.get('sigma_p_plus', 0.0)), 0.01, key=f'p_spp_{stamp}',
                                    help='σp⁺ for para, σm for meta substituents.')
            ald_match = known_ald_smiles.get(Chem.CanonSmiles(ald_smiles)) if parse_structure(ald_smiles) is not None else None
            p_r = R_OF[ald_match] if ald_match else None
            p_label = info.get('label') or 'custom'
            ald_name = ald_match or 'custom aldehyde'
            ald_valid, ald_manual = bool(info.get('scaffold')), not info.get('ok')
        else:
            p_r = R_OF[ald_pick]
            ald_name = ald_pick
            ald_smiles = data.loc[data.aldehyde_name == ald_pick, 'aldehyde_SMILES'].iloc[0] if SMILES_COLUMNS.issubset(data.columns) else None
            ald_valid, ald_manual = True, False
            p_label, p_sp, p_spp = sub_label(p_r), HAMMETT_SIGMA_P[p_r], HAMMETT_SIGMA_P_PLUS[p_r]
            st.caption(f'R = {p_label} · σp {p_sp:+.2f} · σp⁺ {p_spp:+.2f}'.replace('-', '−'))
    with i2:
        amine_pick = st.selectbox('Amine', [a for a in amine_options if a in TAFT_ES] + [CUSTOM], key='p_amine')
        if amine_pick == CUSTOM:
            amine_smiles = structure_input('amine', 'NCC(C)C', 'isobutylamine')
            info = describe_amine(amine_smiles) if amine_smiles else {'ok': False, 'messages': ['Enter a structure.']}
            if amine_smiles and parse_structure(amine_smiles) is not None:
                st.markdown(structure_figure(amine_smiles, info.get('label') or ''), unsafe_allow_html=True)
            for message in info['messages']:
                (st.caption if info['ok'] else st.warning)(message)
            stamp = abs(hash(amine_smiles)) % 10 ** 8
            p_es = st.number_input('Taft Eₛ of the N-alkyl group', -3.0, 1.0, float(info.get('taft_Es', -0.5)), 0.01, key=f'p_es_{stamp}',
                                   help='Filled in from the structure when the alkyl group is recognised; edit to override. More negative = bulkier.')
            amine_match = known_amine_smiles.get(Chem.CanonSmiles(amine_smiles)) if parse_structure(amine_smiles) is not None else None
            amine_name = amine_match or (info.get('label', 'custom') + 'amine' if info.get('ok') else 'custom amine')
            amine_valid, amine_manual = bool(info.get('scaffold')), not info.get('ok')
        else:
            amine_match = amine_pick
            amine_name = amine_pick
            amine_smiles = data.loc[data.amine_name == amine_pick, 'amine_SMILES'].iloc[0] if SMILES_COLUMNS.issubset(data.columns) else None
            p_es = TAFT_ES[amine_pick]
            amine_valid, amine_manual = True, False
            st.caption(f'Taft Eₛ {p_es:+.2f}'.replace('-', '−'))
    c1, c2, c3 = st.columns(3)
    p_solvent = c1.selectbox('Solvent', sorted(data.solvent.unique()), key='p_solvent',
                             help='Only solvents in the training data can be predicted.')
    p_temp = c2.slider('Temperature (°C)', 0.0, 60.0, 25.0, 1.0, key='p_temp',
                       help=f'Measured range {T_LO:g}–{T_HI:g} °C. Outside it the prediction is an extrapolation.')
    p_time = c3.slider('Reaction time (min)', 1.0, 240.0, 60.0, 1.0, key='p_time',
                       help=f'Measured up to {TIME_HI:g} min.')
    c1, c2 = st.columns([2, 1])
    p_model = c1.radio('Model', ['Kinetic GP (recommended)', 'Direct GBM', 'Compare both'], horizontal=True, key='p_model',
                       help='Kinetic GP: Gaussian process on ln k + rate law, with 95% interval. Direct GBM: gradient-boosted trees on conversion.')
    p_target = c2.slider('Target conversion for "time to" (%)', 10, 99, 90, key='p_target')

    inputs_ok = ald_valid and amine_valid
    if not inputs_ok:
        st.info('Enter a valid aryl aldehyde (CHO on a benzene ring) and a primary alkylamine to get a prediction.')
    else:
        feat = np.array([[p_sp, p_spp, p_es, 1.0 if p_solvent.upper() == 'MTBE' else 0.0, 1000 / (p_temp + 273.15)]])
        centre, low, high, ln_k, ln_k_sd = predict_conversion_gp(gp_model, feat, [p_time], ORDER)
        k_pred = float(np.exp(ln_k[0]))
        gbm_at_t = float(predict_conversion_gbm(gbm_model, feat, [p_time])[0, 0])
        eps = 2.5
        lk_lo, _ = predict_ln_k(gp_model, np.array([[*feat[0, :4], 1000 / (p_temp - eps + 273.15)]]))
        lk_hi, _ = predict_ln_k(gp_model, np.array([[*feat[0, :4], 1000 / (p_temp + eps + 273.15)]]))
        ea_pred = -R_GAS * (lk_hi[0] - lk_lo[0]) / (1 / (p_temp + eps + 273.15) - 1 / (p_temp - eps + 273.15)) / 1000

        # Applicability domain
        issues = []
        if not T_LO <= p_temp <= T_HI:
            issues.append(f'temperature {p_temp:g} °C is outside the measured {T_LO:g}–{T_HI:g} °C')
        if p_time > TIME_HI:
            issues.append(f'time {p_time:g} min is beyond the longest measured time ({TIME_HI:g} min)')
        if not min(HAMMETT_SIGMA_P.values()) <= p_sp <= max(HAMMETT_SIGMA_P.values()):
            issues.append('σp is outside the range of the training aldehydes')
        if not min(HAMMETT_SIGMA_P_PLUS.values()) <= p_spp <= max(HAMMETT_SIGMA_P_PLUS.values()):
            issues.append('σp⁺ is outside the range of the training aldehydes')
        if not min(TAFT_ES.values()) <= p_es <= max(TAFT_ES.values()):
            issues.append('Eₛ is outside the range of the training amines')
        measured_pair = p_r is not None and amine_match in TAFT_ES
        novel = not measured_pair

        show_gp = p_model != 'Direct GBM'
        show_gbm = p_model != 'Kinetic GP (recommended)'
        m = st.columns(int(show_gp) + int(show_gbm) + 2)  # one card per shown result, no empty gaps
        slot = iter(m)
        if show_gp:
            gp_card = next(slot)
            gp_card.metric(f'Conversion at {p_time:g} min', f'{centre[0, 0]:.1f}%', help='Kinetic GP prediction')
            gp_card.caption(f'95% interval {low[0, 0]:.1f}–{high[0, 0]:.1f}%')
        if show_gbm:
            next(slot).metric(f'Direct GBM at {p_time:g} min', f'{gbm_at_t:.1f}%')
        t_target = float(time_to_reach(p_target / 100, k_pred, ORDER))
        next(slot).metric(f'Time to {p_target}%', f'{t_target:,.0f} min' if t_target < 1e5 else '> 10⁵ min', help='Kinetic GP + rate law')
        k_card = next(slot)
        k_card.metric('Predicted k_app', f'{k_pred:.3g} min⁻¹', help=f'Apparent Eₐ from the model ≈ {ea_pred:.0f} kJ/mol')
        k_card.caption(f'Eₐ ≈ {ea_pred:.0f} kJ/mol · t½ ≈ {float(time_to_reach(0.5, k_pred, ORDER)):,.0f} min')

        if issues:
            st.warning('Extrapolation: ' + '; '.join(issues) + '. Treat the numbers as a hypothesis to test, not a result.')
        elif novel:
            manual = ' Some descriptors were entered by hand, so check them.' if ald_manual or amine_manual else ''
            st.info('Inside the descriptor range, but this reactant was not measured. Leave-one-aldehyde-out validation suggests '
                    'typical errors of about ±10–15 percentage points for a new substrate.' + manual)
        else:
            st.success('Inside the measured domain. Cross-validated error for new conditions is well under 1 percentage point.')

        product_smiles = imine_product(ald_smiles, amine_smiles) if ald_smiles and amine_smiles else None
        if product_smiles:
            scheme([('mol', ald_smiles, ald_name), ('op', '+'), ('mol', amine_smiles, amine_name),
                    ('arrow', f'{p_solvent}, {p_temp:g} °C', f'{p_time:g} min · predicted {centre[0, 0]:.0f}%'),
                    ('mol', product_smiles, 'imine product'), ('op', '+'), ('mol', 'O', 'water')])
            product = Chem.MolFromSmiles(product_smiles)
            st.caption(f'Product: {rdMolDescriptors.CalcMolFormula(product)}, MW {Descriptors.MolWt(product):.2f} g/mol, '
                       f'SMILES {product_smiles}')

        t_grid = np.linspace(0, max(p_time, TIME_HI) * 1.15, 150)
        fig = go.Figure()
        if show_gp:
            c_mid, c_lo, c_hi, _, _ = predict_conversion_gp(gp_model, feat, t_grid, ORDER)
            fig.add_scatter(x=np.r_[t_grid, t_grid[::-1]], y=np.r_[c_hi[0], c_lo[0][::-1]], fill='toself', line=dict(width=0),
                            fillcolor='rgba(122,184,245,0.22)', name='GP 95% interval', hoverinfo='skip')
            fig.add_scatter(x=t_grid, y=c_mid[0], mode='lines', line=dict(color='#7AB8F5', width=3), name='Kinetic GP')
        if show_gbm:
            fig.add_scatter(x=t_grid, y=predict_conversion_gbm(gbm_model, feat, t_grid)[0], mode='lines',
                            line=dict(color='#E8B04B', width=2, dash='dash'), name='Direct GBM')
        if measured_pair:
            measured = data[(data.aldehyde_name == ald_name) & (data.amine_name == amine_match) & (data.solvent == p_solvent)
                            & (data.temperature_C == p_temp)]
            if not measured.empty:
                fig.add_scatter(x=measured.time_min, y=measured.conversion_percent, mode='markers', name='Measured',
                                marker=dict(size=11, color='#E8615C', symbol='diamond'))
        fig.add_vrect(x0=TIME_HI, x1=t_grid.max(), fillcolor='rgba(127,127,127,0.12)', line_width=0)
        fig.add_vline(x=p_time, line_dash='dot', line_color='gray')
        fig.update_layout(xaxis_title='Time (min)', yaxis_title='Conversion (%)', hovermode='x unified',
                          legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0), margin=dict(t=80))
        fig.update_yaxes(range=[0, 102])
        a, b = st.columns(2)
        with a:
            st.markdown('**Predicted time course**')
            show(fig, 'predicted_time_course', height=380)
            st.caption('Grey band: beyond the longest measured time.')
        with b:
            st.markdown(f'**Temperature response at {p_time:g} min**')
            temp_grid = np.linspace(0, 60, 61)
            feats = np.column_stack([np.repeat(feat[:, :4], len(temp_grid), axis=0), 1000 / (temp_grid + 273.15)])
            fig = go.Figure()
            if show_gp:
                c_mid, c_lo, c_hi, _, _ = predict_conversion_gp(gp_model, feats, [p_time], ORDER)
                fig.add_scatter(x=np.r_[temp_grid, temp_grid[::-1]], y=np.r_[c_hi[:, 0], c_lo[::-1, 0]], fill='toself',
                                line=dict(width=0), fillcolor='rgba(122,184,245,0.22)', name='GP 95% interval', hoverinfo='skip')
                fig.add_scatter(x=temp_grid, y=c_mid[:, 0], mode='lines', line=dict(color='#7AB8F5', width=3), name='Kinetic GP')
            if show_gbm:
                fig.add_scatter(x=temp_grid, y=predict_conversion_gbm(gbm_model, feats, [p_time])[:, 0], mode='lines',
                                line=dict(color='#E8B04B', width=2, dash='dash'), name='Direct GBM')
            fig.add_vrect(x0=T_LO, x1=T_HI, fillcolor='rgba(124,199,154,0.12)', line_width=0)
            fig.add_vline(x=p_temp, line_dash='dot', line_color='gray')
            fig.update_layout(xaxis_title='Temperature (°C)', yaxis_title='Conversion (%)', hovermode='x unified',
                              legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0), margin=dict(t=80))
            fig.update_yaxes(range=[0, 102])
            show(fig, 'predicted_temperature_response', height=380)
            st.caption('Green band: measured temperature range.' + (' Note how the GBM goes flat outside it: trees cannot extrapolate.'
                                                                    if show_gbm else ''))

    with st.expander('Batch predictions from a CSV'):
        st.markdown('Columns: `solvent`, `temperature_C`, `time_min`; the aldehyde as `aldehyde_SMILES`, `aldehyde_R_group` or `sigma_p` '
                    '(optionally `sigma_p_plus`); the amine as `amine_SMILES`, `amine_name` or `taft_Es`.')
        template = pd.DataFrame({'aldehyde_SMILES': ['', '', 'O=Cc1ccc(Br)cc1', ''], 'aldehyde_R_group': ['H', 'NO2', '', ''],
                                 'sigma_p': ['', '', '', 0.06], 'sigma_p_plus': ['', '', '', -0.07],
                                 'amine_SMILES': ['', '', 'NCC(C)C', ''], 'amine_name': ['n-butylamine', 'tert-butylamine', '', ''],
                                 'taft_Es': ['', '', '', -0.47],
                                 'solvent': ['DCM', 'MTBE', 'DCM', 'MTBE'], 'temperature_C': [25, 35, 20, 30], 'time_min': [60, 80, 45, 30]})
        st.download_button('Download template', template.to_csv(index=False).encode(), file_name='prediction_template.csv', mime='text/csv')
        batch_file = st.file_uploader('Upload conditions', type='csv', key='batch_file')
        if batch_file is not None:
            batch = pd.read_csv(batch_file)
            try:
                b = batch.copy()
                # Structures given as SMILES are converted to descriptors first.
                if 'aldehyde_SMILES' in b:
                    described = [describe_aldehyde(x) if isinstance(x, str) and x.strip() else {} for x in b.aldehyde_SMILES]
                    for col, key in (('sigma_p', 'sigma_p'), ('sigma_p_plus', 'sigma_p_plus')):
                        derived = pd.Series([d.get(key) if d.get('ok') else np.nan for d in described], index=b.index, dtype=float)
                        b[col] = b[col].fillna(derived) if col in b else derived
                if 'amine_SMILES' in b:
                    described = [describe_amine(x) if isinstance(x, str) and x.strip() else {} for x in b.amine_SMILES]
                    derived = pd.Series([d.get('taft_Es') if d.get('ok') else np.nan for d in described], index=b.index, dtype=float)
                    b['taft_Es'] = b['taft_Es'].fillna(derived) if 'taft_Es' in b else derived
                if 'sigma_p' not in b or b.sigma_p.isna().any():
                    b['sigma_p'] = b.get('sigma_p', pd.Series(np.nan, index=b.index)).fillna(b.aldehyde_R_group.map(HAMMETT_SIGMA_P))
                if 'sigma_p_plus' not in b or b.sigma_p_plus.isna().any():
                    b['sigma_p_plus'] = b.get('sigma_p_plus', pd.Series(np.nan, index=b.index)).fillna(
                        b.get('aldehyde_R_group', pd.Series(dtype=str)).map(HAMMETT_SIGMA_P_PLUS)).fillna(b.sigma_p)
                if 'taft_Es' not in b or b.taft_Es.isna().any():
                    b['taft_Es'] = b.get('taft_Es', pd.Series(np.nan, index=b.index)).fillna(b.amine_name.map(TAFT_ES))
                b = add_descriptors(b, HAMMETT_SIGMA_P, HAMMETT_SIGMA_P_PLUS, TAFT_ES)
                ok = b[DESCRIPTORS + ['time_min']].notna().all(axis=1)
                X = b.loc[ok, DESCRIPTORS].to_numpy(float)
                out = batch.copy()
                for col in ('gp_conversion_percent', 'gp_low_95', 'gp_high_95', 'gbm_conversion_percent', 'k_app_per_min'):
                    out[col] = np.nan
                if ok.any():
                    mean, sd = predict_ln_k(gp_model, X)
                    t = b.loc[ok, 'time_min'].to_numpy(float)
                    out.loc[ok, 'gp_conversion_percent'] = conversion(t, np.exp(mean), ORDER) * 100
                    out.loc[ok, 'gp_low_95'] = conversion(t, np.exp(mean - 1.96 * sd), ORDER) * 100
                    out.loc[ok, 'gp_high_95'] = conversion(t, np.exp(mean + 1.96 * sd), ORDER) * 100
                    out.loc[ok, 'gbm_conversion_percent'] = np.clip(gbm_model.predict(b.loc[ok, DESCRIPTORS + ['time_min']].to_numpy(float)), 0, 100)
                    out.loc[ok, 'k_app_per_min'] = np.exp(mean)
                if (~ok).any():
                    st.warning(f'{int((~ok).sum())} row(s) lacked descriptors or conditions and were left blank.')
                st.dataframe(out, hide_index=True, width='stretch')
                st.download_button('Download predictions', out.to_csv(index=False).encode(), file_name='imine_predictions.csv', mime='text/csv')
            except Exception as error:  # show a readable message rather than a stack trace
                st.error(f'Could not read that file: {error}')

    with st.expander('How the models work and how well they validate'):
        st.markdown(f'''
**Descriptors.** Aldehyde: σp and σp⁺ of the para-substituent. Amine: Taft Eₛ of the N-alkyl group. Solvent: DCM / MTBE indicator.
Temperature: 1000/T. Time enters only through the rate law (GP) or as a feature (GBM).

**Kinetic GP (recommended).** The {len(kin_desc)} fitted rate constants (Kinetics tab, order n = {ORDER:.2f}) are learned as ln k_app by a
Gaussian process (anisotropic RBF + white-noise kernel, standardised inputs). Conversion follows from the rate law, so curves are smooth,
start at 0, never exceed 100% and carry a 95% interval from the GP variance (parameter uncertainty only, not rate-law error).

**Direct GBM.** Histogram gradient-boosted trees predict conversion (%) from the descriptors and time, constrained to increase with time
and temperature. It needs no kinetic model but is step-like and flat outside the training range.
''')
        if st.toggle('Run cross-validation (takes ~15 s the first time)', key='run_cv'):
            cv_table, cv_preds = prediction_cv(data, ORDER)
            st.dataframe(cv_table, hide_index=True, width='stretch', column_config={
                'RMSE (pp)': st.column_config.NumberColumn(format='%.2f'), 'MAE (pp)': st.column_config.NumberColumn(format='%.2f'),
                'R²': st.column_config.NumberColumn(format='%.3f')})
            scheme_pick = st.radio('Parity plot for', cv_table.Validation.unique().tolist(), horizontal=True, key='cv_scheme')
            sub = cv_preds[cv_preds.Validation == scheme_pick]
            fig = px.scatter(sub, x='observed', y='predicted', color='Model', opacity=0.65,
                             color_discrete_map={'Kinetic GP': '#7AB8F5', 'Direct GBM': '#E8B04B'},
                             hover_data=['aldehyde_R_group', 'amine_name'],
                             labels={'observed': 'Measured conversion (%)', 'predicted': 'Out-of-sample prediction (%)'})
            fig.add_scatter(x=[0, 100], y=[0, 100], mode='lines', line=dict(color='gray', dash='dash'), showlegend=False)
            show(fig, 'parity_plot', height=420)
            st.caption('Random 5-fold = new conditions for known reactants. Leave one aldehyde out = a substrate the model has never seen; '
                       'this is the honest guide to accuracy for custom reactants.')
        ls = gp_model[-1].kernel_.k1.k2.length_scale
        rel = pd.DataFrame({'Descriptor': ['σp', 'σp⁺', 'Taft Eₛ', 'Solvent (MTBE)', '1000/T'], 'Length scale': ls,
                            'Relevance (1/length scale)': 1 / np.asarray(ls)})
        st.markdown('**What the GP relies on.** Shorter length scale (higher relevance) = ln k changes faster with that descriptor.')
        st.dataframe(rel, hide_index=True, width='stretch', column_config={
            'Length scale': st.column_config.NumberColumn(format='%.2f'),
            'Relevance (1/length scale)': st.column_config.ProgressColumn(format='%.2f', min_value=0,
                                                                           max_value=float((1 / np.asarray(ls)).max()))})

with tab1:
    st.subheader('How conversion changes over time')
    st.caption('Colour = temperature, line style = solvent. Each line averages the selected aldehydes and amines. '
               'Use the sidebar to isolate a specific reaction.')
    traces = filtered.groupby(['time_min', 'temperature_C', 'solvent'], as_index=False)['conversion_percent'].mean()
    temp_values = sorted(traces.temperature_C.unique())
    traces['Temperature'] = traces.temperature_C.map(lambda t: f'{t:g} °C')
    temp_labels = [f'{t:g} °C' for t in temp_values]
    # Cool-to-warm colours, so colder runs read as blue and hotter runs as red.
    warm = ['#5B9BF0', '#E8B04B', '#E8615C'] if len(temp_values) <= 3 else \
        px.colors.sample_colorscale('Turbo', [i / (len(temp_values) - 1) for i in range(len(temp_values))])
    solvent_values = sorted(traces.solvent.unique())
    fig = px.line(traces, x='time_min', y='conversion_percent', color='Temperature', line_dash='solvent',
                  symbol='solvent', markers=True,
                  category_orders={'Temperature': temp_labels, 'solvent': solvent_values},
                  color_discrete_sequence=warm[-len(temp_values):] if len(temp_values) < 3 else warm,
                  line_dash_sequence=['solid', 'dash', 'dot', 'dashdot'],
                  symbol_sequence=['circle', 'diamond', 'square', 'triangle-up'],
                  labels={'time_min': 'Time (min)', 'conversion_percent': 'Mean conversion (%)', 'solvent': 'Solvent'},
                  hover_data={'conversion_percent': ':.1f'})
    fig.update_traces(line_width=3, marker_size=9)
    # Label each line at its last point so the legend is not needed to tell them apart.
    ends = traces.sort_values('time_min').groupby(['solvent', 'Temperature'], as_index=False).last()
    color_of = dict(zip(temp_labels, warm[-len(temp_values):] if len(temp_values) < 3 else warm))
    top = traces.conversion_percent.max()
    y_max = min(105, (int(top // 10) + 2) * 10)
    # Spread labels that would overlap: keep at least ~4% of the axis between neighbours.
    ends = ends.sort_values('conversion_percent').reset_index(drop=True)
    gap, label_y = 0.045 * y_max, []
    for value in ends.conversion_percent:
        label_y.append(max(value, label_y[-1] + gap) if label_y else value)
    for end, y in zip(ends.itertuples(), label_y):
        fig.add_annotation(x=end.time_min, y=y, text=f'{end.solvent} {end.Temperature}',
                           showarrow=False, xanchor='left', xshift=10, font=dict(color=color_of[end.Temperature], size=12))
    times_shown = sorted(traces.time_min.unique())
    span = max(times_shown) - min(times_shown) or 10
    fig.update_yaxes(range=[0, max(y_max, label_y[-1] + gap)], gridcolor='rgba(127,127,127,0.2)')
    fig.update_xaxes(tickvals=times_shown, range=[min(times_shown) - 0.05 * span, max(times_shown) + 0.28 * span])
    fig.update_layout(height=500, legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0, title_text=''),
                      hovermode='x unified', margin=dict(t=70))
    show(fig, 'time_courses', height=500)
    st.caption('The y-axis starts at 0 and is scaled to the data shown, so the lines are easier to tell apart.')
    st.caption('Only sampled time points are connected; lines do not establish a kinetic model.')


with tab_k:
    st.subheader('Which rate law describes the data?')
    st.caption('Every time course (aldehyde × amine × solvent × temperature) is fitted to dx/dt = k_app(1 − x)ⁿ by least squares on '
               'conversion. Fits always use all measured times, whatever the time filter.')
    left, right = st.columns([3, 2])
    with left:
        st.dataframe(model_table.assign(**{'In use': model_table['Order n'].sub(ORDER).abs().lt(1e-9).map({True: '✓', False: ''})}),
                     hide_index=True, width='stretch', column_config={
                         'Order n': st.column_config.NumberColumn(format='%.2f'),
                         'RMSE (pp)': st.column_config.NumberColumn('RMSE (pp)', format='%.2f', help='Root-mean-square residual, percentage points'),
                         'AIC': st.column_config.NumberColumn(format='%.0f'),
                         'ΔAIC': st.column_config.NumberColumn(format='%.0f', help='0 = best; > 10 = essentially no support')})
        st.markdown(f'Best shared order **n = {best_order:.2f}** (95% CI {order_ci[0]:.2f}–{order_ci[1]:.2f}). '
                    f'Currently using **{model_name.lower()}** (n = {ORDER:.2f}); change it under *Analysis settings*.')
    with right:
        fig = px.line(order_profile_df, x='n', y='sse', labels={'n': 'Reaction order n', 'sse': 'Total squared residual'})
        fig.add_vline(x=best_order, line_dash='dash', line_color='#E8615C')
        fig.update_yaxes(type='log', dtick=1, exponentformat='power')
        show(fig, 'order_profile', height=260)
    with st.expander('Method and assumptions'):
        st.markdown(f'''
- **Rate law.** x is fractional conversion of the limiting reagent. n = 1 gives x = 1 − e^(−k_app·t); for n ≠ 1,
  x = 1 − [1 + (n − 1)·k_app·t]^(1/(1−n)). With equimolar reagents k_app = k·c₀ⁿ⁻¹, so k_app is in min⁻¹ whatever n is.
- **Fitting.** k_app for each time course by Gauss–Newton least squares on x; n is shared by all {len(kin_all)} time courses and chosen by a profile scan.
  Standard errors come from the Jacobian with m − 1 degrees of freedom.
- **Model choice.** AIC = N ln(SSE/N) + 2p. A non-integer order is an empirical description (for example, mixed rate-limiting
  steps or partial reversibility); it is not a mechanism by itself.
- **Not modelled.** Reverse reaction / equilibrium position, water or acid catalysis, and concentration effects (only one c₀ was run).
- **Units.** Rate constants are reported as {K_UNIT}{' after dividing by c₀ⁿ⁻¹' if c0 > 0 and abs(ORDER - 1) > 1e-9 else ''}.
''')

    st.subheader('Inspect a time course')
    c1, c2, c3, c4 = st.columns([3, 2, 2, 2])
    ins_ald = c1.selectbox('Aldehyde', ald_by_sigma, key='ins_ald')
    ins_amine = c2.selectbox('Amine', amines, key='ins_amine')
    ins_solvent = c3.selectbox('Solvent', solvents, key='ins_solvent')
    view = c4.radio('View', ['Conversion', 'Linearised'], horizontal=True,
                    help='Linearised plots the integrated rate law g(x) against t: a straight line through the origin if the model holds.')
    series = data[(data.aldehyde_name == ins_ald) & (data.amine_name == ins_amine) & (data.solvent == ins_solvent)
                  & data.temperature_C.isin(temperatures)]
    fits = kin_all[(kin_all.aldehyde_name == ins_ald) & (kin_all.amine_name == ins_amine) & (kin_all.solvent == ins_solvent)
                   & kin_all.temperature_C.isin(temperatures)].sort_values('temperature_C')
    t_grid = np.linspace(0, series.time_min.max() * 1.05, 120)
    fig, resid_fig = go.Figure(), go.Figure()
    for colour, fit in zip(TEMP_COLORS * 3, fits.itertuples()):
        pts = series[series.temperature_C == fit.temperature_C].sort_values('time_min')
        x_obs = pts.conversion_percent.to_numpy() / 100
        name = f'{fit.temperature_C:g} °C'
        if view == 'Conversion':
            y_obs, y_fit = x_obs * 100, conversion(t_grid, fit.k_app_per_min, ORDER) * 100
        else:
            y_obs, y_fit = linearised(x_obs, ORDER), fit.k_app_per_min * t_grid
        fig.add_scatter(x=pts.time_min, y=y_obs, mode='markers', marker=dict(size=10, color=colour), name=name, legendgroup=name)
        fig.add_scatter(x=t_grid, y=y_fit, mode='lines', line=dict(color=colour, width=2), name=f'{name} fit',
                        legendgroup=name, showlegend=False, hoverinfo='skip')
        resid = (x_obs - conversion(pts.time_min.to_numpy(), fit.k_app_per_min, ORDER)) * 100
        resid_fig.add_scatter(x=pts.time_min, y=resid, mode='markers+lines', marker=dict(size=8, color=colour),
                              line=dict(color=colour, width=1, dash='dot'), name=name, showlegend=False)
    fig.update_layout(xaxis_title='Time (min)', yaxis_title='Conversion (%)' if view == 'Conversion' else f'g(x), n = {ORDER:.2f}',
                      hovermode='x unified')
    resid_fig.add_hline(y=0, line_color='gray', line_dash='dash')
    resid_fig.update_layout(xaxis_title='Time (min)', yaxis_title='Residual (pp)')
    a, b = st.columns([3, 2])
    with a:
        show(fig, 'time_course_fit', height=380)
    with b:
        show(resid_fig, 'residuals', height=380)
    if not fits.empty:
        st.dataframe(fits.assign(T=fits.temperature_C.map(lambda t: f'{t:g} °C'))[['T', 'k_app_per_min', 'k_app_se', 'k_per_s', 'rmse_pp']],
                     hide_index=True, width='stretch', column_config={
                         'T': 'Temperature', 'k_app_per_min': st.column_config.NumberColumn('k_app (min⁻¹)', format='%.4g'),
                         'k_app_se': st.column_config.NumberColumn('± SE', format='%.2g'),
                         'k_per_s': st.column_config.NumberColumn(f'k ({K_UNIT})', format='%.4g'),
                         'rmse_pp': st.column_config.NumberColumn('RMSE (pp)', format='%.2f')})

    st.subheader('Condition planner')
    st.caption('Arrhenius interpolation of k_app for the time course chosen above, then the rate law solved for time. '
               'Temperatures are limited to the measured range.')
    temps_avail = sorted(kin_all[(kin_all.aldehyde_name == ins_ald) & (kin_all.amine_name == ins_amine)
                                 & (kin_all.solvent == ins_solvent)].temperature_C.unique())
    arr = ols(1 / (np.array(temps_avail) + 273.15), np.log(
        kin_all.set_index(['aldehyde_name', 'amine_name', 'solvent', 'temperature_C'])
        .loc[[(ins_ald, ins_amine, ins_solvent, t) for t in temps_avail], 'k_app_per_min'].to_numpy()))
    if arr is None:
        st.info('Needs at least two measured temperatures for this reaction.')
    else:
        p1, p2 = st.columns(2)
        target = p1.slider('Target conversion (%)', 5, 99, 90)
        plan_t = p2.slider('Temperature (°C)', float(min(temps_avail)), float(max(temps_avail)),
                           float(np.median(temps_avail)), step=1.0)
        k_plan = float(np.exp(arr['intercept'] + arr['slope'] / (plan_t + 273.15)))
        t_needed = float(time_to_reach(target / 100, k_plan, ORDER))
        t_max = data.time_min.max()
        m1, m2, m3 = st.columns(3)
        m1.metric('Interpolated k_app', f'{k_plan:.3g} min⁻¹')
        m2.metric(f'Time to {target}%', f'{t_needed:,.0f} min' if t_needed < 1e5 else '> 10⁵ min')
        m3.metric(f'Conversion at {t_max:g} min', f'{conversion(t_max, k_plan, ORDER) * 100:.1f}%')
        if t_needed > t_max:
            st.warning(f'{t_needed:,.0f} min is beyond the longest measured time ({t_max:g} min): this is an extrapolation of the rate law.')
        temp_grid = np.linspace(min(temps_avail), max(temps_avail), 41)
        k_grid = np.exp(arr['intercept'] + arr['slope'] / (temp_grid + 273.15))
        fig = go.Figure(go.Scatter(x=temp_grid, y=time_to_reach(target / 100, k_grid, ORDER), mode='lines',
                                   line=dict(width=3, color='#7AB8F5'), name='time needed'))
        fig.add_hline(y=t_max, line_dash='dash', line_color='gray')
        # Annotations on a log axis take log10 coordinates.
        fig.add_annotation(x=max(temps_avail), y=np.log10(t_max), text='longest measured time', showarrow=False,
                           xanchor='right', yanchor='bottom', font=dict(size=11, color='gray'))
        fig.add_scatter(x=[plan_t], y=[t_needed], mode='markers', marker=dict(size=12, color='#E8615C'), name='chosen')
        fig.update_layout(xaxis_title='Temperature (°C)', yaxis_title=f'Time to {target}% (min)', showlegend=False)
        fig.update_yaxes(type='log', dtick='D2')
        show(fig, 'condition_planner', height=320)

    st.subheader('Rate-constant map')
    if kin.empty:
        st.info('No time courses match the filters.')
    else:
        m1, m2 = st.columns(2)
        map_solvent = m1.selectbox('Solvent', sorted(kin.solvent.unique()), key='map_solvent')
        map_temp = m2.selectbox('Temperature', sorted(kin.temperature_C.unique()), key='map_temp', format_func=lambda t: f'{t:g} °C')
        sub = kin[(kin.solvent == map_solvent) & (kin.temperature_C == map_temp)]
        grid = sub.pivot_table(index='aldehyde_R_group', columns='amine_name', values='k_app_per_min')
        grid = grid.reindex([r for r in R_GROUP_ORDER if r in grid.index]).reindex(columns=[a for a in amine_options if a in grid.columns])
        fig = px.imshow(np.log10(grid), color_continuous_scale=RATE_SCALE, aspect='auto',
                        labels={'x': 'Amine', 'y': 'para-R (donating → withdrawing)', 'color': 'log₁₀ k_app'})
        fig.update_traces(text=grid.map(lambda v: f'{v:.2e}').to_numpy(), texttemplate='%{text}',
                          hovertemplate='R = %{y}<br>%{x}<br>k_app = %{text} min⁻¹<extra></extra>')
        fig.update_traces(xgap=2, ygap=2)
        show(fig, 'rate_constant_map', height=420)
        st.caption('Colour is log₁₀ k_app, so equal colour steps are equal factors in rate. Numbers are k_app in min⁻¹.')
    with st.expander('All fitted rate constants (current filters)'):
        table = kin[['aldehyde_name', 'aldehyde_R_group', 'amine_name', 'solvent', 'temperature_C',
                     'k_app_per_min', 'k_app_se', 'k_per_s', 'rmse_pp']].sort_values(['amine_name', 'solvent', 'temperature_C', 'aldehyde_R_group'])
        st.dataframe(table, hide_index=True, width='stretch', column_config={
            'k_app_per_min': st.column_config.NumberColumn('k_app (min⁻¹)', format='%.4g'),
            'k_app_se': st.column_config.NumberColumn('± SE', format='%.2g'),
            'k_per_s': st.column_config.NumberColumn(f'k ({K_UNIT})', format='%.4g'),
            'rmse_pp': st.column_config.NumberColumn('RMSE (pp)', format='%.2f')})
        st.download_button('Download rate constants (CSV)', table.assign(order_n=ORDER).to_csv(index=False).encode(),
                           file_name='imine_rate_constants.csv', mime='text/csv')

with tab_h:
    st.subheader(f'Hammett plot ({sigma_name})')
    if kin.empty:
        st.info('No time courses match the filters.')
    else:
        h1, h2, h3 = st.columns([2, 2, 3])
        h_solvent = h1.selectbox('Solvent', sorted(kin.solvent.unique()), key='h_solvent')
        h_temp = h2.selectbox('Temperature', sorted(kin.temperature_C.unique()), key='h_temp', format_func=lambda t: f'{t:g} °C')
        present = [r for r in R_GROUP_ORDER if r in set(kin.aldehyde_R_group)]
        excluded = h3.multiselect('Exclude from the fit', present, format_func=sub_label,
                                  help='Excluded points stay on the plot as hollow markers. Useful for testing curvature or outliers.')
        k_h = kin_all[kin_all.aldehyde_R_group == 'H'].set_index(['amine_name', 'solvent', 'temperature_C']).k_app_per_min

        def hammett_points(frame):
            frame = frame[frame.aldehyde_R_group.isin(SIGMA)].copy()
            frame['sigma'] = frame.aldehyde_R_group.map(SIGMA)
            ref = [k_h.get((a, s, t), np.nan) for a, s, t in zip(frame.amine_name, frame.solvent, frame.temperature_C)]
            frame['log_rel'] = np.log10(frame.k_app_per_min / np.array(ref))
            return frame

        pts = hammett_points(kin[(kin.solvent == h_solvent) & (kin.temperature_C == h_temp)])
        fig, rows = go.Figure(), []
        s_range = np.linspace(min(SIGMA.values()) - 0.1, max(SIGMA.values()) + 0.1, 2)
        for amine in [a for a in amine_options if a in set(pts.amine_name)]:
            part = pts[pts.amine_name == amine]
            used = part[~part.aldehyde_R_group.isin(excluded)]
            colour = AMINE_COLORS.get(amine, '#999')
            fig.add_scatter(x=used.sigma, y=used.log_rel, mode='markers', customdata=used.aldehyde_R_group.map(sub_label),
                            hovertemplate='R = %{customdata}<br>σ = %{x:+.2f}<br>log(k/k_H) = %{y:.2f}<extra>' + amine + '</extra>',
                            marker=dict(size=11, color=colour), name=amine, legendgroup=amine)
            out = part[part.aldehyde_R_group.isin(excluded)]
            if not out.empty:
                fig.add_scatter(x=out.sigma, y=out.log_rel, mode='markers', customdata=out.aldehyde_R_group.map(sub_label),
                                hovertemplate='R = %{customdata} (excluded)<br>σ = %{x:+.2f}<br>log(k/k_H) = %{y:.2f}<extra></extra>',
                                marker=dict(size=11, color='rgba(0,0,0,0)', line=dict(color=colour, width=2)),
                                name=f'{amine} (excluded)', legendgroup=amine, showlegend=False)
            fit = ols(used.sigma, used.log_rel)
            if fit:
                fig.add_scatter(x=s_range, y=fit['intercept'] + fit['slope'] * s_range, mode='lines',
                                line=dict(color=colour, width=2, dash='dash'), showlegend=False, hoverinfo='skip')
                rows.append({'Amine': amine, 'ρ': fit['slope'], '± SE': fit['se_slope'], 'Intercept': fit['intercept'],
                             'R²': fit['r2'], 'Points': fit['n']})
        shown_r = [r for r in R_GROUP_ORDER if r in set(pts.aldehyde_R_group)]
        fig.update_layout(xaxis_title=sigma_name, yaxis_title='log₁₀(k_X / k_H)', hovermode='closest')
        fig.update_xaxes(tickvals=[SIGMA[r] for r in shown_r], tickangle=0,
                         ticktext=[f'{sub_label(r)}<br>{SIGMA[r]:+.2f}'.replace('-', '−') for r in shown_r])
        show(fig, 'hammett', height=460)
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch', column_config={
                c: st.column_config.NumberColumn(format='%.2f') for c in ('ρ', '± SE', 'Intercept')} | {
                'R²': st.column_config.NumberColumn(format='%.3f')})
        st.caption('ρ > 0: electron-withdrawing groups accelerate the reaction, so negative charge builds up (or positive charge is lost) '
                   'at the carbonyl carbon in the rate-limiting step. Curvature or a poor linear fit can signal a change in rate-limiting '
                   'step across the series (addition vs. dehydration). Switch between σp and σp⁺ under Analysis settings to compare.')

        a, b = st.columns(2)
        with a:
            st.markdown('**ρ across conditions**')
            rho_rows = []
            for (amine, solvent, temp), part in hammett_points(kin).groupby(['amine_name', 'solvent', 'temperature_C']):
                fit = ols(part[~part.aldehyde_R_group.isin(excluded)].sigma, part[~part.aldehyde_R_group.isin(excluded)].log_rel)
                if fit:
                    rho_rows.append({'amine': amine, 'solvent': solvent, 'T': temp, 'rho': fit['slope'], 'se': fit['se_slope']})
            if rho_rows:
                rho = pd.DataFrame(rho_rows)
                fig = px.line(rho, x='T', y='rho', color='amine', line_dash='solvent', markers=True, error_y='se',
                              color_discrete_map=AMINE_COLORS, labels={'T': 'Temperature (°C)', 'rho': 'ρ'})
                fig.update_xaxes(tickvals=sorted(rho['T'].unique()))
                fig.update_traces(error_y=dict(thickness=1, width=0))
                show(fig, 'rho_vs_temperature', height=340)
        with b:
            st.markdown('**Which σ scale fits better?**')
            cmp_rows = []
            for scale_name, scale in (('σp', HAMMETT_SIGMA_P), ('σp⁺', HAMMETT_SIGMA_P_PLUS)):
                for (amine, solvent, temp), part in kin.groupby(['amine_name', 'solvent', 'temperature_C']):
                    part = part[part.aldehyde_R_group.isin(scale) & ~part.aldehyde_R_group.isin(excluded)]
                    fit = ols(part.aldehyde_R_group.map(scale), np.log10(part.k_app_per_min))
                    if fit:
                        cmp_rows.append({'Scale': scale_name, 'Amine': amine, 'R²': fit['r2']})
            if cmp_rows:
                cmp = pd.DataFrame(cmp_rows).pivot_table(index='Amine', columns='Scale', values='R²', aggfunc='mean')
                cmp = cmp.reindex([a for a in amine_options if a in cmp.index])
                st.dataframe(cmp, width='stretch', column_config={c: st.column_config.NumberColumn(f'mean R² ({c})', format='%.3f')
                                                                  for c in cmp.columns})
                st.caption('Mean R² of the linear Hammett fit over all filtered solvent/temperature combinations. '
                           'A clearly better σp⁺ fit points to through-conjugation with a cationic centre (e.g. an iminium-like transition state).')

    st.subheader('Taft steric analysis')
    if kin.empty or kin.amine_name.nunique() < 2:
        st.info('Select at least two amines in the sidebar.')
    else:
        t1, t2 = st.columns(2)
        ta_solvent = t1.selectbox('Solvent', sorted(kin.solvent.unique()), key='t_solvent')
        ta_temp = t2.selectbox('Temperature', sorted(kin.temperature_C.unique()), key='t_temp', format_func=lambda t: f'{t:g} °C')
        part = kin[(kin.solvent == ta_solvent) & (kin.temperature_C == ta_temp) & kin.amine_name.isin(TAFT_ES)].copy()
        part['Es'] = part.amine_name.map(TAFT_ES)
        k_ref = part.set_index(['aldehyde_name']).query('amine_name == "n-butylamine"').k_app_per_min
        part['log_rel'] = np.log10(part.k_app_per_min / part.aldehyde_name.map(k_ref))
        fig, rows = go.Figure(), []
        es_range = np.linspace(min(TAFT_ES.values()) - 0.1, max(TAFT_ES.values()) + 0.1, 2)
        for r in [r for r in R_GROUP_ORDER if r in set(part.aldehyde_R_group)]:
            sub = part[part.aldehyde_R_group == r].sort_values('Es')
            fit = ols(sub.Es, sub.log_rel)
            colour = SUB_COLORS.get(r, '#999')
            fig.add_scatter(x=sub.Es, y=sub.log_rel, mode='markers', marker=dict(size=10, color=colour), name=f'R = {sub_label(r)}')
            if fit:
                fig.add_scatter(x=es_range, y=fit['intercept'] + fit['slope'] * es_range, mode='lines',
                                line=dict(color=colour, width=1.5, dash='dash'), showlegend=False, hoverinfo='skip')
                rows.append({'Aldehyde R': sub_label(r), 'δ': fit['slope'], '± SE': fit['se_slope'], 'R²': fit['r2']})
        fig.update_layout(xaxis_title='Taft Eₛ of the N-alkyl group (more negative = bulkier)', yaxis_title='log₁₀(k / k_n-Bu)')
        fig.update_xaxes(tickvals=list(TAFT_ES.values()), ticktext=[f'{v:+.2f}<br>{k.split("-")[0]}-Bu' for k, v in TAFT_ES.items()])
        a, b = st.columns([3, 2])
        with a:
            show(fig, 'taft', height=400)
        with b:
            if rows:
                st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch', column_config={
                    c: st.column_config.NumberColumn(format='%.2f') for c in ('δ', '± SE')} | {'R²': st.column_config.NumberColumn(format='%.3f')})
            st.caption('log(k/k_ref) = δ·Eₛ. A larger δ means the reaction is more sensitive to steric bulk next to nitrogen. '
                       'Only three amines, so each fit has one degree of freedom: treat δ as indicative.')

with tab_e:
    st.subheader('Eyring and Arrhenius analysis')
    if act.empty:
        st.info('Select at least two temperatures (and some reactants) in the sidebar.')
    else:
        e1, e2, e3 = st.columns([2, 2, 2])
        plot_kind = e1.radio('Plot', ['Eyring', 'Arrhenius'], horizontal=True)
        e_amine = e2.selectbox('Amine', [a for a in amine_options if a in set(act.amine_name)], key='e_amine')
        e_solvent = e3.selectbox('Solvent', sorted(act.solvent.unique()), key='e_solvent')
        part = kin[(kin.amine_name == e_amine) & (kin.solvent == e_solvent)].copy()
        part['inv_T'] = 1000 / (part.temperature_C + 273.15)
        part['y'] = np.log(part.k_per_s / (part.temperature_C + 273.15)) if plot_kind == 'Eyring' else np.log(part.k_per_s)
        fig = go.Figure()
        for r in [r for r in R_GROUP_ORDER if r in set(part.aldehyde_R_group)]:
            sub = part[part.aldehyde_R_group == r].sort_values('inv_T')
            fit = ols(sub.inv_T, sub.y)
            colour = SUB_COLORS.get(r, '#999')
            fig.add_scatter(x=sub.inv_T, y=sub.y, mode='markers', marker=dict(size=10, color=colour), name=f'R = {sub_label(r)}',
                            legendgroup=r, customdata=sub.temperature_C, hovertemplate='%{customdata} °C<br>y = %{y:.3f}<extra></extra>')
            if fit:
                xs = np.array([sub.inv_T.min(), sub.inv_T.max()])
                fig.add_scatter(x=xs, y=fit['intercept'] + fit['slope'] * xs, mode='lines', line=dict(color=colour, width=2),
                                showlegend=False, legendgroup=r, hoverinfo='skip')
        fig.update_layout(xaxis_title='1000 / T (K⁻¹)',
                          yaxis_title=f'ln(k / T)  [k in {K_UNIT}]' if plot_kind == 'Eyring' else f'ln k  [k in {K_UNIT}]')
        show(fig, f'{plot_kind.lower()}_plot', height=420)
        st.caption('Eyring: ln(k/T) = ln(k_B/h) + ΔS‡/R − ΔH‡/(RT). Arrhenius: ln k = ln A − Eₐ/(RT). '
                   f'Each line uses {int(act.n_temps.max())} temperatures, so uncertainties rest on '
                   f'{int(act.n_temps.max()) - 2} degree(s) of freedom.')
        if abs(ORDER - 1) > 1e-9 and c0 <= 0:
            st.info('ΔS‡ (and ln A) depend on the units of k. With n ≠ 1 and no c₀ they refer to k_app, so compare them only with each other. '
                    'Enter c₀ under Analysis settings for concentration-based values. Eₐ and ΔH‡ are unaffected.')

        st.markdown('**Activation parameters**')
        table = act.sort_values(['amine_name', 'solvent', 'aldehyde_R_group'], key=lambda c: c.map(
            {r: i for i, r in enumerate(R_GROUP_ORDER)}) if c.name == 'aldehyde_R_group' else c)
        table = table.assign(**{'R': table.aldehyde_R_group.map(sub_label)})[
            ['R', 'amine_name', 'solvent', 'Ea_kJ', 'Ea_se', 'dH_kJ', 'dH_se', 'dS_J', 'dS_se', 'dG298_kJ', 'r2']]
        st.dataframe(table, hide_index=True, width='stretch', column_config={
            'amine_name': 'Amine', 'solvent': 'Solvent',
            'Ea_kJ': st.column_config.NumberColumn('Eₐ (kJ/mol)', format='%.1f'),
            'Ea_se': st.column_config.NumberColumn('± SE', format='%.1f'),
            'dH_kJ': st.column_config.NumberColumn('ΔH‡ (kJ/mol)', format='%.1f'),
            'dH_se': st.column_config.NumberColumn('± SE', format='%.1f'),
            'dS_J': st.column_config.NumberColumn('ΔS‡ (J/mol·K)', format='%.0f'),
            'dS_se': st.column_config.NumberColumn('± SE', format='%.0f'),
            'dG298_kJ': st.column_config.NumberColumn('ΔG‡₂₉₈ (kJ/mol)', format='%.1f'),
            'r2': st.column_config.NumberColumn('R² (Arrhenius)', format='%.4f')})
        st.download_button('Download activation parameters (CSV)', act.assign(order_n=ORDER, k_units=K_UNIT).to_csv(index=False).encode(),
                           file_name='imine_activation_parameters.csv', mime='text/csv')

        st.markdown('**Enthalpy–entropy compensation**')
        comp = ols(act.dS_J, act.dH_kJ * 1000)
        fig = px.scatter(act.assign(R=act.aldehyde_R_group.map(sub_label)), x='dS_J', y='dH_kJ', color='amine_name', symbol='solvent',
                         hover_data={'R': True}, color_discrete_map=AMINE_COLORS,
                         labels={'dS_J': 'ΔS‡ (J/mol·K)', 'dH_kJ': 'ΔH‡ (kJ/mol)', 'amine_name': 'Amine', 'solvent': 'Solvent'})
        fig.update_traces(marker_size=11)
        if comp:
            xs = np.array([act.dS_J.min(), act.dS_J.max()])
            fig.add_scatter(x=xs, y=(comp['intercept'] + comp['slope'] * xs) / 1000, mode='lines',
                            line=dict(color='gray', dash='dash'), name='linear fit')
        show(fig, 'compensation', height=380)
        if comp:
            st.caption(f'Slope (isokinetic temperature β) = {comp["slope"]:.0f} ± {comp["se_slope"]:.0f} K, R² = {comp["r2"]:.2f}. '
                       'ΔH‡ and ΔS‡ come from the same fit, so their errors are correlated and can create an apparent compensation line; '
                       'test it against the spread of ΔG‡ before interpreting β (Exner, Prog. Phys. Org. Chem. 1973, 10, 411).')

with tab2:
    st.subheader('Aldehyde substituent × amine')
    st.caption('Cells show mean conversion across the currently selected times, temperatures and solvents.')
    heat = filtered.pivot_table(index='aldehyde_R_group', columns='amine_name',
                                values='conversion_percent', aggfunc='mean')
    order = [r for r in R_GROUP_ORDER if r in heat.index] + [r for r in heat.index if r not in R_GROUP_ORDER]
    heat = heat.reindex(order)
    fig = px.imshow(heat, text_auto='.1f', color_continuous_scale=CONVERSION_SCALE, zmin=0, zmax=100,
                    labels={'x': 'Amine', 'y': 'Aldehyde R group (donating → withdrawing)', 'color': 'Conversion (%)'},
                    aspect='auto')
    fig.update_traces(xgap=2, ygap=2)
    show(fig, 'reactivity_map', height=460)
    st.caption('For fairer comparisons, select one temperature, one time and one solvent in the sidebar.')

with tab3:
    st.subheader('Compare like-for-like experiments')
    with st.expander('How this comparison works'):
        st.markdown('''
Averages across the whole dataset can mislead, because the groups being compared may differ in other ways too.
This tab avoids that by **pairing experiments that are identical except for the one variable you choose**.

1. Pick the **variable** to change and the two **levels** to compare: a baseline (A) and a comparison (B).
2. Every B experiment is paired with the A experiment that matches it on **all other recorded variables**
   (aldehyde, amine, solvent, temperature and time, minus the one you are changing).
3. **Conversion** compares each pair as B − A in percentage points. **Rate constant** compares the fitted k_app of whole
   time courses as the ratio k_B / k_A, which avoids the ceiling at 100% conversion.
4. The cards summarise the pairs: mean with 95% confidence interval, median, how often B beats A, and a Wilcoxon
   signed-rank test of whether the differences are centred on zero.

**Reading the charts.** The histogram shows how consistent the effect is. In the paired plot every point is one pair;
points above the diagonal are pairs where B is higher. The breakdown heatmap shows where the effect is large or small
(red = B higher, blue = A higher). *All comparisons* at the bottom ranks every pair of levels for every variable.

Differences are in percentage points, not percent change (40% → 50% is +10 pp). Only experiments allowed by the sidebar
filters are paired. Each pair is one combination of conditions, not an independent replicate, so treat p-values as descriptive.
''')

    MC_KEYS = ['aldehyde_R_group', 'amine_name', 'solvent', 'temperature_C', 'time_min']
    MC_VARIABLES = {'Aldehyde substituent': 'aldehyde_R_group', 'Amine': 'amine_name', 'Solvent': 'solvent',
                    'Temperature': 'temperature_C', 'Reaction time': 'time_min'}
    MC_NAMES = {v: k for k, v in MC_VARIABLES.items()}

    def mc_order(col, values):
        values = list(dict.fromkeys(values))
        if col == 'aldehyde_R_group':
            return sorted(values, key=lambda r: (HAMMETT_SIGMA_P.get(r, 99), r))
        if col == 'amine_name':
            return sorted(values, key=lambda a: (list(AMINE_NOTES).index(a) if a in AMINE_NOTES else 99, a))
        return sorted(values)

    def mc_label(col, value):
        if col == 'aldehyde_R_group':
            return f'R = {sub_label(value)}'
        if col == 'temperature_C':
            return f'{value:g} °C'
        if col == 'time_min':
            return f'{value:g} min'
        return str(value)

    def mc_pairs(frame, col, a, b, value):
        keys = [k for k in MC_KEYS if k in frame.columns and k != col]
        table = frame[frame[col].isin([a, b])].pivot_table(index=keys, columns=col, values=value, aggfunc='mean')
        if a not in table.columns or b not in table.columns:
            return pd.DataFrame()
        table = table[[a, b]].dropna().reset_index()
        table.columns = keys + ['A', 'B']
        return table

    def mc_stats(diff):
        diff = np.asarray(diff, float)
        n = len(diff)
        out = {'n': n, 'mean': diff.mean() if n else np.nan, 'median': np.median(diff) if n else np.nan,
               'q1': np.percentile(diff, 25) if n else np.nan, 'q3': np.percentile(diff, 75) if n else np.nan,
               'lo': np.nan, 'hi': np.nan, 'b_wins': (diff > 0).mean() if n else np.nan,
               'a_wins': (diff < 0).mean() if n else np.nan, 'p': np.nan}
        if n >= 2:
            half = scipy_stats.t.ppf(0.975, n - 1) * diff.std(ddof=1) / np.sqrt(n)
            out['lo'], out['hi'] = out['mean'] - half, out['mean'] + half
        if n >= 6 and np.any(diff != 0):
            try:
                out['p'] = float(scipy_stats.wilcoxon(diff).pvalue)
            except ValueError:
                pass
        return out

    def fmt_p(p):
        return '—' if not np.isfinite(p) else ('< 0.001' if p < 0.001 else f'{p:.3f}')

    st.markdown('#### Single comparison')
    c1, c2, c3, c4 = st.columns([2, 2, 2, 2])
    mc_var_name = c1.selectbox('Variable to change', list(MC_VARIABLES), index=2, key='mc_var')
    mc_col = MC_VARIABLES[mc_var_name]
    mc_levels = mc_order(mc_col, filtered[mc_col].unique())
    if len(mc_levels) < 2:
        st.info(f'Only one {mc_var_name.lower()} is selected in the sidebar. Select at least two to compare.')
    else:
        mc_a = c2.selectbox('Baseline (A)', mc_levels, index=0, format_func=lambda v: mc_label(mc_col, v), key=f'mc_a_{mc_col}')
        b_options = [v for v in mc_levels if v != mc_a]
        mc_b = c3.selectbox('Compared with (B)', b_options, index=len(b_options) - 1, format_func=lambda v: mc_label(mc_col, v),
                            key=f'mc_b_{mc_col}')
        rate_ok = mc_col != 'time_min'
        mc_measure = c4.radio('Measure', ['Conversion', 'Rate constant'] if rate_ok else ['Conversion'], horizontal=True, key='mc_measure',
                              help='Rate constant compares fitted k_app of whole time courses (Kinetics tab), so time is not a variable there.')
        use_rate = mc_measure == 'Rate constant'
        source = kin if use_rate else filtered
        pairs = mc_pairs(source, mc_col, mc_a, mc_b, 'k_app_per_min' if use_rate else 'conversion_percent')
        la, lb = mc_label(mc_col, mc_a), mc_label(mc_col, mc_b)
        if pairs.empty:
            st.info('No matched pairs remain under these filters.')
        else:
            if use_rate:
                pairs['effect'] = np.log10(pairs.B / pairs.A)
                stt = mc_stats(pairs.effect)
                fold = lambda v: 10 ** v
                m = st.columns(5)
                m[0].metric('Matched pairs', f'{stt["n"]}', help='Time courses matched on every other variable')
                m[1].metric('Geometric-mean k_B / k_A', f'{fold(stt["mean"]):.3g}×')
                m[1].caption(f'95% CI {fold(stt["lo"]):.3g}–{fold(stt["hi"]):.3g}×')
                m[2].metric('Median k_B / k_A', f'{fold(stt["median"]):.3g}×')
                m[2].caption(f'IQR {fold(stt["q1"]):.3g}–{fold(stt["q3"]):.3g}×')
                m[3].metric(f'{lb} faster in', f'{stt["b_wins"] * 100:.0f}%', help='Share of matched time courses where k_B > k_A')
                m[3].caption(f'of {stt["n"]} pairs')
                m[4].metric('Wilcoxon p', fmt_p(stt['p']), help='Signed-rank test on log(k_B/k_A); tests whether the ratio is centred on 1.')
                direction = 'faster' if stt['mean'] > 0 else 'slower'
                st.markdown(f'With **{lb}** the reaction is on average **{fold(abs(stt["mean"])):.3g}× {direction}** '
                            f'than with **{la}**, and {lb} is faster in **{stt["b_wins"] * 100:.0f}%** of {stt["n"]} matched time courses.')
            else:
                pairs['effect'] = pairs.B - pairs.A
                stt = mc_stats(pairs.effect)
                m = st.columns(5)
                m[0].metric('Matched pairs', f'{stt["n"]}', help='Measurements matched on every other variable')
                m[1].metric('Mean difference', f'{stt["mean"]:+.1f} pp')
                m[1].caption(f'95% CI {stt["lo"]:+.1f} to {stt["hi"]:+.1f} pp')
                m[2].metric('Median difference', f'{stt["median"]:+.1f} pp')
                m[2].caption(f'IQR {stt["q1"]:+.1f} to {stt["q3"]:+.1f} pp')
                m[3].metric(f'{lb} higher in', f'{stt["b_wins"] * 100:.0f}%', help='Share of matched pairs where B > A')
                m[3].caption(f'{la} higher in {stt["a_wins"] * 100:.0f}%; ties {max(0.0, 100 - (stt["b_wins"] + stt["a_wins"]) * 100):.0f}%')
                m[4].metric('Wilcoxon p', fmt_p(stt['p']), help='Signed-rank test of whether the paired differences are centred on zero.')
                st.markdown(f'Switching from **{la}** to **{lb}** changes conversion by **{stt["mean"]:+.1f} percentage points** on average '
                            f'(95% CI {stt["lo"]:+.1f} to {stt["hi"]:+.1f}); {lb} gives higher conversion in **{stt["b_wins"] * 100:.0f}%** '
                            f'of {stt["n"]} matched pairs.')

            others = [k for k in MC_KEYS if k in pairs.columns]
            g1, g2 = st.columns(2)
            with g1:
                st.markdown('**Distribution of paired effects**')
                xlab = f'log₁₀(k_B / k_A)' if use_rate else 'B − A (percentage points)'
                fig = px.histogram(pairs, x='effect', nbins=30, labels={'effect': xlab}, color_discrete_sequence=['#7AB8F5'])
                fig.update_traces(marker_line_width=0.5, marker_line_color='rgba(0,0,0,0.3)')
                fig.add_vline(x=0, line_dash='dash', line_color='gray')
                fig.add_vline(x=stt['mean'], line_color='#E8615C', line_width=2,
                              annotation_text='mean', annotation_font_color='#E8615C', annotation_position='top')
                fig.update_yaxes(title_text='Matched pairs')
                show(fig, 'matched_distribution', height=380)
            with g2:
                st.markdown('**Paired values**')
                colour_by = st.selectbox('Colour points by', [MC_NAMES[k] for k in others], key='mc_colour',
                                         label_visibility='collapsed')
                ccol = MC_VARIABLES[colour_by]
                plot = pairs.assign(group=pairs[ccol].map(lambda v: mc_label(ccol, v)))
                order = [mc_label(ccol, v) for v in mc_order(ccol, pairs[ccol].unique())]
                fig = px.scatter(plot, x='A', y='B', color='group', category_orders={'group': order}, opacity=0.8,
                                 color_discrete_map=({mc_label(ccol, r): SUB_COLORS.get(r, '#999') for r in pairs[ccol].unique()}
                                                     if ccol == 'aldehyde_R_group' else
                                                     {a: AMINE_COLORS.get(a, '#999') for a in pairs[ccol].unique()} if ccol == 'amine_name' else None),
                                 labels={'A': f'{la} ({"k_app, min⁻¹" if use_rate else "conversion, %"})',
                                         'B': f'{lb} ({"k_app, min⁻¹" if use_rate else "conversion, %"})', 'group': colour_by})
                lo_v = min(pairs.A.min(), pairs.B.min())
                hi_v = max(pairs.A.max(), pairs.B.max())
                fig.add_scatter(x=[lo_v, hi_v], y=[lo_v, hi_v], mode='lines', line=dict(color='gray', dash='dash'),
                                name='no change', hoverinfo='skip')
                fig.update_traces(marker_size=8, selector=dict(mode='markers'))
                if use_rate:
                    fig.update_xaxes(type='log')
                    fig.update_yaxes(type='log')
                show(fig, 'matched_paired', height=340)

            if len(others) >= 1:
                st.markdown('**Where is the effect largest?**')
                b1, b2 = st.columns(2)
                row_name = b1.selectbox('Rows', [MC_NAMES[k] for k in others], index=0, key='mc_rows')
                col_choices = ['(none)'] + [MC_NAMES[k] for k in others if MC_NAMES[k] != row_name]
                col_name = b2.selectbox('Columns', col_choices, index=1 if len(col_choices) > 1 else 0, key='mc_cols')
                rcol = MC_VARIABLES[row_name]
                if col_name == '(none)':
                    grid = pairs.groupby(rcol).effect.mean().to_frame('mean effect')
                    grid.index = [mc_label(rcol, v) for v in grid.index]
                    grid = grid.reindex([mc_label(rcol, v) for v in mc_order(rcol, pairs[rcol].unique())])
                else:
                    ccol2 = MC_VARIABLES[col_name]
                    grid = pairs.pivot_table(index=rcol, columns=ccol2, values='effect', aggfunc='mean')
                    grid = grid.reindex(index=mc_order(rcol, grid.index), columns=mc_order(ccol2, grid.columns))
                    grid.index = [mc_label(rcol, v) for v in grid.index]
                    grid.columns = [mc_label(ccol2, v) for v in grid.columns]
                bound = float(np.nanmax(np.abs(grid.to_numpy()))) or 1.0
                fig = px.imshow(grid, color_continuous_scale=BLUE_RED_SCALE, zmin=-bound, zmax=bound, aspect='auto',
                                text_auto='.2f' if use_rate else '+.1f',
                                labels={'color': 'mean log₁₀(k_B/k_A)' if use_rate else 'mean B − A (pp)'})
                fig.update_traces(xgap=2, ygap=2)
                show(fig, 'matched_breakdown', height=max(260, 48 * len(grid) + 120))
                st.caption(f'Each cell is the mean paired effect of {la} → {lb} for that subset. Red: {lb} higher; blue: {la} higher; '
                           'white: no change. The colour range is symmetric about zero.')

            with st.expander(f'See all {len(pairs)} matched pairs'):
                shown_pairs = pairs.rename(columns={'A': la, 'B': lb, 'effect': 'log10(k_B/k_A)' if use_rate else 'B − A (pp)'})
                st.dataframe(shown_pairs, hide_index=True, width='stretch')
                st.download_button('Download matched pairs (CSV)', shown_pairs.to_csv(index=False).encode(),
                                   file_name=f'matched_{mc_col}_{mc_a}_vs_{mc_b}.csv', mime='text/csv')

    st.markdown('#### All comparisons')
    st.caption('Every pair of levels for every variable, matched on all other variables, using the current sidebar filters. '
               'Conversion effects in percentage points; rate effects as the geometric-mean k ratio.')
    rows = []
    for name, col in MC_VARIABLES.items():
        levels = mc_order(col, filtered[col].unique())
        for i, a in enumerate(levels):
            for b in levels[i + 1:]:
                cp = mc_pairs(filtered, col, a, b, 'conversion_percent')
                if cp.empty:
                    continue
                cs = mc_stats(cp.B - cp.A)
                rp = mc_pairs(kin, col, a, b, 'k_app_per_min') if col in kin.columns and col != 'time_min' else pd.DataFrame()
                rs = mc_stats(np.log10(rp.B / rp.A)) if not rp.empty else None
                rows.append({'Variable': name, 'A': mc_label(col, a), 'B': mc_label(col, b), 'Pairs': cs['n'],
                             'Mean B − A (pp)': cs['mean'], 'CI low': cs['lo'], 'CI high': cs['hi'],
                             'B higher (%)': cs['b_wins'] * 100, 'p (Wilcoxon)': cs['p'],
                             'k_B / k_A': 10 ** rs['mean'] if rs else np.nan, 'adjacent': b == levels[i + 1]})
    if not rows:
        st.info('Select at least two levels of some variable in the sidebar.')
    else:
        overview = pd.DataFrame(rows)
        f1, f2 = st.columns([3, 2])
        show_vars = f1.pills('Variables', list(MC_VARIABLES), selection_mode='multi', default=list(MC_VARIABLES), key='mc_overview_vars')
        adjacent_only = f2.toggle('Adjacent levels only', value=False, key='mc_adjacent',
                                  help='e.g. 15→25 and 25→35 °C, but not 15→35 °C. Substituents are ordered by σp, amines by bulk.')
        view = overview[overview.Variable.isin(show_vars or [])]
        if adjacent_only:
            view = view[view.adjacent]
        if view.empty:
            st.info('Nothing to show: choose at least one variable.')
        else:
            view = view.iloc[::-1]
            labels = view.A + ' → ' + view.B
            var_colors = dict(zip(MC_VARIABLES, ['#E8615C', '#B48CE0', '#7CC79A', '#E8B04B', '#7AB8F5']))
            fig = go.Figure()
            for name in MC_VARIABLES:
                part = view[view.Variable == name]
                if part.empty:
                    continue
                fig.add_scatter(x=part['Mean B − A (pp)'], y=(part.A + ' → ' + part.B), mode='markers', name=name,
                                marker=dict(size=10, color=var_colors[name]),
                                error_x=dict(type='data', symmetric=False, array=part['CI high'] - part['Mean B − A (pp)'],
                                             arrayminus=part['Mean B − A (pp)'] - part['CI low'], thickness=1.5, width=0),
                                customdata=np.column_stack([part.Pairs, part['B higher (%)']]),
                                hovertemplate='%{y}<br>mean %{x:+.1f} pp<br>%{customdata[0]} pairs, B higher in %{customdata[1]:.0f}%<extra></extra>')
            fig.add_vline(x=0, line_dash='dash', line_color='gray')
            fig.update_layout(xaxis_title='Mean paired difference, B − A (percentage points, 95% CI)',
                              yaxis=dict(categoryorder='array', categoryarray=list(labels)),
                              legend=dict(orientation='h', yanchor='bottom', y=1.01, x=0), margin=dict(t=70))
            show(fig, 'matched_overview_forest', height=max(320, 26 * len(view) + 140))
            table = view.iloc[::-1].drop(columns='adjacent')
            st.dataframe(table, hide_index=True, width='stretch', column_config={
                'Mean B − A (pp)': st.column_config.NumberColumn(format='%+.1f'),
                'CI low': st.column_config.NumberColumn('95% CI low', format='%+.1f'),
                'CI high': st.column_config.NumberColumn('95% CI high', format='%+.1f'),
                'B higher (%)': st.column_config.ProgressColumn('B higher', format='%.0f%%', min_value=0, max_value=100),
                'p (Wilcoxon)': st.column_config.NumberColumn(format='%.2g'),
                'k_B / k_A': st.column_config.NumberColumn('k_B / k_A (geo-mean)', format='%.3g')})
            st.download_button('Download all comparisons (CSV)', table.to_csv(index=False).encode(),
                               file_name='matched_comparisons_overview.csv', mime='text/csv')

with tab4:
    st.subheader('Find an observed condition')
    shown = filtered.sort_values('conversion_percent', ascending=False)
    st.dataframe(shown, hide_index=True, width='stretch')
    st.download_button('Download filtered CSV', shown.to_csv(index=False).encode('utf-8'),
                       file_name='filtered_imine_reactions.csv', mime='text/csv')
    st.markdown('''**Tabs at a glance**

| Tab | Use it to |
|---|---|
| Reaction scheme | See the reaction, any specific example, and the substrate set with σ / Eₛ values |
| Predict | ML predictions of conversion, rate and time to target. Pick dataset reactants or enter your own structure (SMILES, drawing or name); batch CSV mode |
| Time courses | Compare averaged conversion curves by temperature and solvent |
| Kinetics | Choose a rate law, inspect fits and residuals, map rate constants, plan conditions |
| Hammett & Taft | Electronic (ρ) and steric (δ) sensitivity, how ρ changes with conditions, σp vs σp⁺ |
| Activation parameters | Eyring / Arrhenius plots, Eₐ, ΔH‡, ΔS‡, ΔG‡ and compensation analysis |
| Reactivity map, Matched comparisons | Model-free views of the raw conversions |

Every chart has a camera button (top right) that saves it as SVG or PNG at 3× resolution; choose the format under *Analysis settings*.

**Reading the dashboard**

- Conversion is the measured percentage at the stated time, not isolated yield.
- The sidebar filters every panel; choosing several settings creates averages only where a panel says so.
- The matched-comparison panel compares measurements at identical values of all *other* recorded experimental variables.
- A high conversion at 80 minutes does not prove that a condition is fast at 20 minutes. Inspect the time course before choosing a lab condition.
- Do not extrapolate to untested substrates, times, temperatures or solvents without further experiments.
''')

st.html('<div class="app-footer"><span>Imine Reaction Explorer</span>'
        '<span>σ constants: Hansch, Leo &amp; Taft, Chem. Rev. 1991, 91, 165</span>'
        '<span>Built with Streamlit, Plotly, RDKit and scikit-learn</span>'
        '<span>Descriptive and predictive summaries; verify before planning experiments</span></div>')
