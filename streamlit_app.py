import base64
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
.gallery {display:flex; flex-wrap:wrap; gap:0.75rem; margin-bottom:1rem;}
.gallery figure {margin:0; padding:0.5rem 0.75rem; border-radius:0.5rem; background:rgba(127,127,127,0.1);
                 display:flex; flex-direction:column; align-items:center; justify-content:flex-end; min-width:9rem;}
.gallery figcaption {font-size:0.82rem; text-align:center;}
.gallery small {opacity:0.75;}
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


def gallery(items):
    """items: (smiles, title, subtitle)."""
    cards = ''.join(f'<figure>{structure_svg(smiles, 22)}<figcaption>{escape(title)}<br><small>{escape(sub)}</small></figcaption></figure>'
                    for smiles, title, sub in items)
    st.markdown(SCHEME_CSS + f'<div class="gallery">{cards}</div>', unsafe_allow_html=True)

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

st.sidebar.header('Filter experiments')
aldehydes = st.sidebar.multiselect('Aldehydes', sorted(data.aldehyde_name.unique()), default=sorted(data.aldehyde_name.unique()))
amines = st.sidebar.multiselect('Amines', sorted(data.amine_name.unique()), default=sorted(data.amine_name.unique()))
solvents = st.sidebar.multiselect('Solvents', sorted(data.solvent.unique()), default=sorted(data.solvent.unique()))
temperatures = st.sidebar.multiselect('Temperature (°C)', sorted(data.temperature_C.unique()), default=sorted(data.temperature_C.unique()))
times = st.sidebar.multiselect('Time (min)', sorted(data.time_min.unique()), default=sorted(data.time_min.unique()))

filtered = data[
    data.aldehyde_name.isin(aldehydes)
    & data.amine_name.isin(amines)
    & data.solvent.isin(solvents)
    & data.temperature_C.isin(temperatures)
    & data.time_min.isin(times)
].copy()
if filtered.empty:
    st.warning('No measurements match these filters. Select at least one value in each filter.')
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
        st.markdown('**Aldehydes**, from electron-donating to electron-withdrawing (Hammett σp)')
        alds = data.drop_duplicates('aldehyde_name')
        alds = alds.assign(order=alds.aldehyde_R_group.map({r: i for i, r in enumerate(R_GROUP_ORDER)})).sort_values('order')
        gallery([(r.aldehyde_SMILES, r.aldehyde_name,
                  f'R = {r.aldehyde_R_group}' + (f', σp = {HAMMETT_SIGMA_P[r.aldehyde_R_group]:+.2f}'
                                                 if r.aldehyde_R_group in HAMMETT_SIGMA_P else ''))
                 for r in alds.itertuples()])
        st.markdown('**Amines**, increasing steric bulk at the carbon next to nitrogen')
        amine_rows = data.drop_duplicates('amine_name').set_index('amine_name')
        gallery([(amine_rows.loc[a, 'amine_SMILES'], a, AMINE_NOTES.get(a, ''))
                 for a in sorted(amine_rows.index, key=lambda n: (list(AMINE_NOTES).index(n) if n in AMINE_NOTES else 99, n))])

with tab1:
    st.subheader('How conversion changes over time')
    st.caption('Each line averages the selected aldehydes and amines at a fixed temperature and solvent. Use the sidebar to isolate a specific reaction.')
    traces = filtered.groupby(['time_min', 'temperature_C', 'solvent'], as_index=False)['conversion_percent'].mean()
    traces['Condition'] = traces.apply(lambda row: f'{row["solvent"]} · {row["temperature_C"]:g} °C', axis=1)
    fig = px.line(traces, x='time_min', y='conversion_percent', color='Condition', markers=True,
                  labels={'time_min': 'Time (min)', 'conversion_percent': 'Mean conversion (%)'},
                  hover_data={'temperature_C': True, 'solvent': True, 'Condition': False})
    fig.update_yaxes(range=[0, 105])
    st.plotly_chart(fig, width='stretch')
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
