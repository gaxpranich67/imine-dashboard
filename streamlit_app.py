from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

CSV_PATH = Path(__file__).with_name('imine_condensation_dataset_for_students.csv')
# Para-substituents ordered by Hammett sigma_p (electron-donating -> electron-withdrawing).
R_GROUP_ORDER = ['NH2', 'OMe', 'CH3', 'H', 'Cl', 'CF3', 'NO2']
REQUIRED = {
    'aldehyde_name', 'aldehyde_R_group', 'amine_name', 'solvent',
    'temperature_C', 'time_min', 'conversion_percent',
}

st.set_page_config(page_title='Imine Reaction Explorer', page_icon='⚗️', layout='wide')
st.title('⚗️ Imine Reaction Explorer')
st.caption('Explore 504 imine-condensation measurements: 7 aldehydes × 3 amines × 2 solvents × 3 temperatures × 4 times.')

@st.cache_data
def read_csv(source):
    return pd.read_csv(source)

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

tab1, tab2, tab3, tab4 = st.tabs(['Time courses', 'Reactivity map', 'Matched comparisons', 'Measurements & guide'])

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
