import base64
import re
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D

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
st.title('Imine Reaction Explorer')

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
    options.updateAtomPalette({0: (0, 0, 0), 7: (0.35, 0.58, 0.98), 8: (0.93, 0.36, 0.34), 9: (0.30, 0.74, 0.45), 17: (0.30, 0.74, 0.45)})
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return svg[svg.index('<svg'):].replace('#000000', 'currentColor').replace('\n', ' ')


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
st.caption(f'{len(data)} imine-condensation measurements: {data.aldehyde_name.nunique()} aldehydes × {data.amine_name.nunique()} amines × '
           f'{data.solvent.nunique()} solvents × {data.temperature_C.nunique()} temperatures × {data.time_min.nunique()} times.')

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


def show(fig, name, height=None):
    """Render a Plotly chart with a consistent look and a publication-quality download button."""
    fig.update_layout(margin=dict(l=10, r=10, b=10, t=max(50, fig.layout.margin.t or 0)), legend_title_text='',
                      font=dict(size=13), **({'height': height} if height else {}))
    fig.update_xaxes(gridcolor='rgba(127,127,127,0.18)', zeroline=False)
    fig.update_yaxes(gridcolor='rgba(127,127,127,0.18)', zeroline=False)
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

tab0, tab1, tab_k, tab_h, tab_e, tab2, tab3, tab4 = st.tabs(['Reaction scheme', 'Time courses', 'Kinetics', 'Hammett & Taft',
                                                         'Activation parameters', 'Reactivity map', 'Matched comparisons', 'Data & guide'])

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

kin = kin_all[kin_all.aldehyde_name.isin(aldehydes) & kin_all.amine_name.isin(amines)
              & kin_all.solvent.isin(solvents) & kin_all.temperature_C.isin(temperatures)].copy()
act = act_all[act_all.aldehyde_name.isin(aldehydes) & act_all.amine_name.isin(amines) & act_all.solvent.isin(solvents)].copy()
ald_by_sigma = sorted(aldehydes, key=lambda a: HAMMETT_SIGMA_P.get(data.loc[data.aldehyde_name == a, 'aldehyde_R_group'].iloc[0], 99))
R_OF = data.drop_duplicates('aldehyde_name').set_index('aldehyde_name').aldehyde_R_group.to_dict()

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
        fig = px.imshow(np.log10(grid), color_continuous_scale='Viridis', aspect='auto',
                        labels={'x': 'Amine', 'y': 'para-R (donating → withdrawing)', 'color': 'log₁₀ k_app'})
        fig.update_traces(text=grid.map(lambda v: f'{v:.2e}').to_numpy(), texttemplate='%{text}',
                          hovertemplate='R = %{y}<br>%{x}<br>k_app = %{text} min⁻¹<extra></extra>')
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
    fig = px.imshow(heat, text_auto='.1f', color_continuous_scale='Viridis', zmin=0, zmax=100,
                    labels={'x': 'Amine', 'y': 'Aldehyde R group (donating → withdrawing)', 'color': 'Conversion (%)'},
                    aspect='auto')
    show(fig, 'reactivity_map', height=460)
    st.caption('For fairer comparisons, select one temperature, one time and one solvent in the sidebar.')

with tab3:
    st.subheader('Compare like-for-like experiments')
    comparison = st.radio('Change one variable while holding the others fixed',
                          ['Solvent: MTBE − DCM', 'Temperature: 35 − 15 °C',
                           'Amine: n-butylamine − tert-butylamine'], horizontal=True)
    if comparison.startswith('Solvent'):
        variable, low, high = 'solvent', 'DCM', 'MTBE'
    elif comparison.startswith('Temperature'):
        variable, low, high = 'temperature_C', 15, 35
    else:
        variable, low, high = 'amine_name', 'tert-butylamine', 'n-butylamine'
    keys = ['aldehyde_name', 'amine_name', 'solvent', 'temperature_C', 'time_min']
    match_keys = [key for key in keys if key != variable]
    pairs = filtered.pivot_table(index=match_keys, columns=variable,
                                 values='conversion_percent', aggfunc='mean')
    if low not in pairs.columns or high not in pairs.columns:
        st.info('Select both comparison groups in the sidebar to see matched pairs.')
    else:
        pairs = pairs[[low, high]].dropna().reset_index()
        pairs.columns = [str(c) for c in pairs.columns]
        low, high = str(low), str(high)
        if pairs.empty:
            st.info('No matched pairs remain under these filters.')
        else:
            pairs['Difference (percentage points)'] = pairs[high] - pairs[low]
            change = pairs['Difference (percentage points)']
            st.metric('Mean matched difference', f'{change.mean():+.1f} percentage points')
            st.caption(f'{len(pairs)} matched pairs. Positive means {high} has higher conversion than {low}; this is not a percent change.')
            fig = px.histogram(pairs, x='Difference (percentage points)', nbins=25,
                               labels={'count': 'Matched pairs'})
            fig.update_yaxes(title_text='Matched pairs')
            fig.add_vline(x=0, line_dash='dash', line_color='gray')
            show(fig, 'matched_differences', height=380)
            with st.expander('See matched experiments'):
                st.dataframe(pairs, hide_index=True, width='stretch')

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
