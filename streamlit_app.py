import base64
import re
from html import escape
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

CSV_PATH = Path(__file__).with_name('imine_condensation_dataset_for_students.csv')
# Para-substituents ordered by Hammett sigma_p (electron-donating -> electron-withdrawing).
R_GROUP_ORDER = ['NH2', 'OMe', 'CH3', 'H', 'Cl', 'CF3', 'NO2']
HAMMETT_SIGMA_P = {'NH2': -0.66, 'OMe': -0.27, 'CH3': -0.17, 'H': 0.00, 'Cl': 0.23, 'CF3': 0.54, 'NO2': 0.78}
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
st.caption('Explore 504 imine-condensation measurements: 7 aldehydes × 3 amines × 2 solvents × 3 temperatures × 4 times.')

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

if CSV_PATH.exists():
    data = read_csv(CSV_PATH)
else:
    uploaded = st.file_uploader('Upload imine_condensation_dataset_for_students.csv', type='csv')
    if uploaded is None:
        st.info('Upload the workshop CSV to begin. You can also change CSV_PATH at the top of this script to point to your local file.')
        st.stop()
    data = read_csv(uploaded)

missing = REQUIRED - set(data.columns)
if missing:
    st.error(f'Missing required columns: {", ".join(sorted(missing))}')
    st.stop()

data = data.copy()
for column in ('temperature_C', 'time_min', 'conversion_percent'):
    data[column] = pd.to_numeric(data[column], errors='coerce')
data = data.dropna(subset=list(REQUIRED))

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

with st.sidebar:
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

tab0, tab1, tab2, tab3, tab4 = st.tabs(['Reaction scheme', 'Time courses', 'Reactivity map', 'Matched comparisons', 'Measurements & guide'])

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

with tab1:
    st.subheader('How conversion changes over time')
    st.caption('One panel per solvent; each line is one temperature, averaged over the selected aldehydes and amines. '
               'Use the sidebar to isolate a specific reaction.')
    traces = filtered.groupby(['time_min', 'temperature_C', 'solvent'], as_index=False)['conversion_percent'].mean()
    temp_values = sorted(traces.temperature_C.unique())
    traces['Temperature'] = traces.temperature_C.map(lambda t: f'{t:g} °C')
    temp_labels = [f'{t:g} °C' for t in temp_values]
    # Cool-to-warm colours, so colder runs read as blue and hotter runs as red.
    warm = ['#5B9BF0', '#E8B04B', '#E8615C'] if len(temp_values) <= 3 else \
        px.colors.sample_colorscale('Turbo', [i / (len(temp_values) - 1) for i in range(len(temp_values))])
    fig = px.line(traces, x='time_min', y='conversion_percent', color='Temperature', facet_col='solvent',
                  markers=True, category_orders={'Temperature': temp_labels},
                  color_discrete_sequence=warm[-len(temp_values):] if len(temp_values) < 3 else warm,
                  labels={'time_min': 'Time (min)', 'conversion_percent': 'Mean conversion (%)', 'solvent': 'Solvent'},
                  hover_data={'solvent': False, 'conversion_percent': ':.1f'})
    fig.for_each_annotation(lambda a: a.update(text=f'<b>{a.text.split("=")[-1]}</b>', font_size=14))
    fig.update_traces(line_width=3, marker_size=9)
    top = traces.conversion_percent.max()
    fig.update_yaxes(range=[0, min(105, (int(top // 10) + 2) * 10)], gridcolor='rgba(127,127,127,0.2)')
    fig.update_xaxes(tickvals=sorted(traces.time_min.unique()))
    fig.update_layout(height=460, legend=dict(orientation='h', yanchor='bottom', y=1.1, xanchor='left', x=0), margin=dict(t=90),
                      hovermode='x unified')
    st.plotly_chart(fig, width='stretch')
    st.caption('The y-axis starts at 0 and is scaled to the data shown, so the lines are easier to tell apart.')
    st.caption('Only sampled time points are connected; lines do not establish a kinetic model.')

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
    st.plotly_chart(fig, width='stretch')
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
            st.plotly_chart(fig, width='stretch')
            with st.expander('See matched experiments'):
                st.dataframe(pairs, hide_index=True, width='stretch')

with tab4:
    st.subheader('Find an observed condition')
    shown = filtered.sort_values('conversion_percent', ascending=False)
    st.dataframe(shown, hide_index=True, width='stretch')
    st.download_button('Download filtered CSV', shown.to_csv(index=False).encode('utf-8'),
                       file_name='filtered_imine_reactions.csv', mime='text/csv')
    st.markdown('''**Reading the dashboard**

- Conversion is the measured percentage at the stated time, not isolated yield.
- The sidebar filters every panel; choosing several settings creates averages only where a panel says so.
- The matched-comparison panel compares measurements at identical values of all *other* recorded experimental variables.
- A high conversion at 80 minutes does not prove that a condition is fast at 20 minutes. Inspect the time course before choosing a lab condition.
- Do not extrapolate to untested substrates, times, temperatures or solvents without further experiments.
''')
