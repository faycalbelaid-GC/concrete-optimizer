"""
Interactive Dash dashboard — Concrete Mix Optimiser & Durability Simulator.

Run:  python dashboard.py
Open: http://127.0.0.1:8050
"""
import numpy as np
import dash
from dash import dcc, html, Input, Output

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from concrete_optimizer.models.mix_design import MixDesign
from concrete_optimizer.models.strength import (
    compressive_strength_28d, concrete_class, characteristic_strength)
from concrete_optimizer.models.carbon_footprint import embodied_co2, CO2_FACTORS
from concrete_optimizer.durability.carbonation import (
    carbonation_coefficient_k, carbonation_depth_array)
from concrete_optimizer.durability.chloride import (
    chloride_profile, chloride_diffusion_coefficient, CS_BY_CLASS)
from concrete_optimizer.durability.service_life import estimate_service_life

# ── App ────────────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    title='BetonOptimizer — Durabilité & CO₂',
    meta_tags=[{'name': 'viewport',
                'content': 'width=device-width, initial-scale=1'}],
)

# ── Layout helpers ─────────────────────────────────────────────────────────────
def _slider(label, id_, min_, max_, step, value, marks_step=None):
    marks = {}
    if marks_step:
        marks = {v: str(v) for v in range(int(min_), int(max_) + 1, marks_step)}
    return html.Div([
        html.Label(label, style={'fontWeight': '600', 'fontSize': '12px',
                                  'color': '#34495e', 'marginTop': '12px'}),
        dcc.Slider(min_, max_, step, value=value, id=id_,
                    marks=marks,
                    tooltip={'placement': 'bottom', 'always_visible': True}),
    ])


def _kpi(title, value_id, unit, color):
    return html.Div([
        html.Div(title, style={'fontSize': '11px', 'color': '#7f8c8d',
                                'fontWeight': '600', 'marginBottom': '4px'}),
        html.Div(id=value_id, style={'fontSize': '26px', 'fontWeight': '700',
                                      'color': color}),
        html.Div(unit, style={'fontSize': '11px', 'color': '#95a5a6'}),
    ], style={
        'background': 'white', 'borderRadius': '8px',
        'padding': '14px 18px', 'minWidth': '120px',
        'boxShadow': '0 2px 6px rgba(0,0,0,0.08)',
        'borderLeft': f'4px solid {color}',
        'flex': '1',
    })


# ── Layout ─────────────────────────────────────────────────────────────────────
app.layout = html.Div([

    # ── Header
    html.Div([
        html.H1('Optimiseur Béton — Durabilité & Empreinte Carbone',
                style={'margin': '0', 'fontSize': '20px', 'color': 'white'}),
        html.P('Carbonatation (Fick 1) · Chlorures (Fick 2) · NSGA-II',
               style={'margin': '2px 0 0', 'color': '#bdc3c7', 'fontSize': '12px'}),
    ], style={
        'background': 'linear-gradient(135deg,#2c3e50,#3498db)',
        'padding': '16px 30px', 'marginBottom': '20px',
    }),

    html.Div([

        # ── Left panel: controls
        html.Div([
            html.H3('Formulation', style={'color': '#2c3e50', 'marginTop': '0'}),
            _slider('Ciment (kg/m³)', 'cement', 200, 550, 10, 350, 100),
            _slider('Eau / liant effectif  (w/b)', 'wb', 0.30, 0.65, 0.01, 0.45),
            _slider('Cendres volantes  (% liant)', 'fa-pct', 0, 50, 1, 0, 10),
            _slider('Laitier GGBS  (% liant)',     'slag-pct', 0, 70, 1, 0, 10),
            _slider('Fumée de silice  (% liant)',  'sf-pct',   0, 15, 1, 0, 5),

            html.Hr(style={'margin': '18px 0'}),
            html.H3("Conditions d'exposition",
                    style={'color': '#2c3e50', 'marginTop': '0'}),

            _slider('Enrobage (mm)', 'cover', 20, 75, 5, 40, 10),
            html.Label('Classe carbonatation',
                        style={'fontWeight': '600', 'fontSize': '12px',
                               'color': '#34495e', 'marginTop': '12px',
                               'display': 'block'}),
            dcc.Dropdown(['XC1', 'XC2', 'XC3', 'XC4'], 'XC3', id='carb-class',
                          clearable=False),
            html.Label('Classe chlorures',
                        style={'fontWeight': '600', 'fontSize': '12px',
                               'color': '#34495e', 'marginTop': '12px',
                               'display': 'block'}),
            dcc.Dropdown(['XS1', 'XS2', 'XS3', 'XD1', 'XD2', 'XD3'], 'XS2',
                          id='cl-class', clearable=False),
            _slider('Humidité relative (%)', 'rh', 40, 95, 5, 70, 10),
        ], style={
            'width': '270px', 'minWidth': '270px',
            'background': '#f8f9fa', 'borderRadius': '10px',
            'padding': '20px', 'marginRight': '20px',
            'boxShadow': '0 2px 8px rgba(0,0,0,0.06)',
        }),

        # ── Right panel: KPIs + tabs
        html.Div([

            # KPI row
            html.Div([
                _kpi("Résistance f'c",  'kpi-fc',      'MPa',       '#e74c3c'),
                _kpi('Empreinte CO₂',   'kpi-co2',     'kg/m³',     '#27ae60'),
                _kpi('Durée de vie',    'kpi-sl',      'ans',       '#2980b9'),
                _kpi('t initiation',    'kpi-tinit',   'ans',       '#f39c12'),
                _kpi('t propagation',   'kpi-tprop',   'ans',       '#8e44ad'),
                _kpi('Classe béton',    'kpi-class',   'EN 206',    '#16a085'),
            ], style={'display': 'flex', 'gap': '12px',
                       'marginBottom': '20px', 'flexWrap': 'wrap'}),

            # Charts
            dcc.Tabs([
                dcc.Tab(label='Carbonatation', children=[
                    dcc.Graph(id='carb-plot', style={'height': '430px'})]),
                dcc.Tab(label='Profils Cl⁻', children=[
                    dcc.Graph(id='chloride-plot', style={'height': '430px'})]),
                dcc.Tab(label='Durée de vie', children=[
                    dcc.Graph(id='sl-plot', style={'height': '430px'})]),
                dcc.Tab(label='Composition', children=[
                    dcc.Graph(id='comp-plot', style={'height': '430px'})]),
                dcc.Tab(label='Sensibilité w/b', children=[
                    dcc.Graph(id='sens-plot', style={'height': '430px'})]),
            ], style={'background': 'white', 'borderRadius': '8px',
                       'boxShadow': '0 2px 8px rgba(0,0,0,0.06)'}),

        ], style={'flex': '1', 'minWidth': '0'}),

    ], style={'display': 'flex', 'padding': '0 20px 20px',
               'alignItems': 'flex-start'}),

], style={'fontFamily': "'Segoe UI', Arial, sans-serif",
           'background': '#ecf0f1', 'minHeight': '100vh'})


# ── Helpers ────────────────────────────────────────────────────────────────────
def _build_mix(cement, wb, fa_pct, slag_pct, sf_pct):
    fa_r  = fa_pct  / 100.0
    sl_r  = slag_pct / 100.0
    sf_r  = sf_pct  / 100.0
    total_scm = fa_r + sl_r + sf_r
    if total_scm > 0.85:
        s = 0.85 / total_scm
        fa_r, sl_r, sf_r = fa_r * s, sl_r * s, sf_r * s

    cement_f  = max(1.0 - fa_r - sl_r - sf_r, 0.15)
    total_b   = cement / cement_f
    fly_ash   = fa_r * total_b
    slag      = sl_r * total_b
    sf        = sf_r * total_b
    b_eff     = cement + 0.4 * fly_ash + 0.6 * slag + 2.0 * sf
    water     = wb * b_eff

    rho = dict(cement=3150, fa=2200, sl=2900, sf=2200,
               water=1000, fine=2650, coarse=2700)
    v_paste = (cement / rho['cement'] + fly_ash / rho['fa']
               + slag / rho['sl']    + sf / rho['sf']
               + water / rho['water'])
    v_agg  = max(1.0 - v_paste - 0.02, 0.30)
    return MixDesign(
        cement=cement, water=water, fly_ash=fly_ash, slag=slag, silica_fume=sf,
        fine_agg=0.45 * v_agg * rho['fine'],
        coarse_agg=0.55 * v_agg * rho['coarse'],
    )


# ── Callback ───────────────────────────────────────────────────────────────────
@app.callback(
    [Output('kpi-fc',      'children'),
     Output('kpi-co2',     'children'),
     Output('kpi-sl',      'children'),
     Output('kpi-tinit',   'children'),
     Output('kpi-tprop',   'children'),
     Output('kpi-class',   'children'),
     Output('carb-plot',   'figure'),
     Output('chloride-plot','figure'),
     Output('sl-plot',     'figure'),
     Output('comp-plot',   'figure'),
     Output('sens-plot',   'figure')],
    [Input('cement',    'value'),
     Input('wb',        'value'),
     Input('fa-pct',    'value'),
     Input('slag-pct',  'value'),
     Input('sf-pct',    'value'),
     Input('cover',     'value'),
     Input('carb-class','value'),
     Input('cl-class',  'value'),
     Input('rh',        'value')],
)
def update(cement, wb, fa_pct, slag_pct, sf_pct,
           cover, carb_class, cl_class, rh_pct):
    rh  = rh_pct / 100.0
    mix = _build_mix(cement, wb, fa_pct, slag_pct, sf_pct)
    fc  = compressive_strength_28d(mix)
    co2 = embodied_co2(mix)
    sl  = estimate_service_life(mix, cover, carb_class, cl_class, rh)
    fck = characteristic_strength(fc)

    t_init = (sl.t_initiation_carbonation
              if sl.governing_mechanism == 'carbonation'
              else sl.t_initiation_chloride)

    # KPIs
    kpis = [
        f"{fc:.1f}", f"{co2:.0f}", f"{sl.t_service_total:.0f}",
        f"{t_init:.0f}", f"{sl.t_propagation:.1f}", concrete_class(fck),
    ]

    # ── Carbonation chart
    t_arr = np.linspace(0, 120, 600)
    k     = carbonation_coefficient_k(mix, rh)
    xc    = carbonation_depth_array(t_arr, k)

    carb_fig = go.Figure()
    carb_fig.add_trace(go.Scatter(
        x=t_arr, y=xc, name=f'k = {k:.2f} mm/an⁰·⁵',
        line=dict(color='#3498db', width=2.5),
        fill='tozeroy', fillcolor='rgba(52,152,219,0.08)',
    ))
    carb_fig.add_hline(y=cover, line_dash='dash', line_color='crimson',
                        annotation_text=f'Enrobage = {cover} mm',
                        annotation_position='bottom right')
    ti_c = sl.t_initiation_carbonation
    if ti_c < 120:
        carb_fig.add_vline(x=ti_c, line_dash='dot', line_color='darkorange',
                            annotation_text=f't_init = {ti_c:.0f} ans',
                            annotation_position='top right')
    carb_fig.update_layout(
        title=f'Carbonatation — classe {carb_class}',
        xaxis_title='Temps (ans)', yaxis_title='Profondeur carbonatation (mm)',
        template='plotly_white', hovermode='x unified',
    )

    # ── Chloride profiles chart
    D_a, m_age = chloride_diffusion_coefficient(mix)
    Cs = CS_BY_CLASS.get(cl_class, 3.0)
    x_d = np.linspace(0, 100, 400)
    t_list = [2, 5, 10, 25, 50, 100]
    blues  = ['#d6eaf8','#a9cce3','#7fb3d3','#5499c2','#2980b9','#1a5276']

    cl_fig = go.Figure()
    for t_y, col in zip(t_list, blues):
        C = chloride_profile(x_d, t_y, D_a, m_age, Cs)
        cl_fig.add_trace(go.Scatter(
            x=x_d, y=C, name=f't = {t_y} ans',
            line=dict(color=col, width=1.8),
        ))
    cl_fig.add_vline(x=cover, line_dash='dash', line_color='crimson',
                      annotation_text=f'Enrobage {cover}mm')
    cl_fig.add_hline(y=0.40, line_dash='dot', line_color='darkorange',
                      annotation_text='C_crit = 0.40%')
    cl_fig.update_layout(
        title=f'Profils Cl⁻ — classe {cl_class}  |  D_a = {D_a:.1f} mm²/an',
        xaxis_title='Profondeur (mm)',
        yaxis_title='Teneur Cl⁻ (% masse ciment)',
        xaxis_range=[0, 100], yaxis_range=[0, Cs * 1.1],
        template='plotly_white', hovermode='x unified',
    )

    # ── Service life decomposition
    mechs  = ['Carbonatation', 'Chlorures']
    t_inits_plot = [
        min(sl.t_initiation_carbonation, 600),
        min(sl.t_initiation_chloride, 600),
    ]
    t_props = [sl.t_propagation, sl.t_propagation]

    sl_fig = go.Figure()
    sl_fig.add_trace(go.Bar(
        name='Initiation', x=mechs, y=t_inits_plot,
        marker_color=['#3498db','#e74c3c'], text=[f'{v:.0f}' for v in t_inits_plot],
        textposition='inside',
    ))
    sl_fig.add_trace(go.Bar(
        name='Propagation', x=mechs, y=t_props,
        marker_color=['#2980b9','#c0392b'], text=[f'{v:.1f}' for v in t_props],
        textposition='inside',
    ))
    gov_idx = 0 if sl.governing_mechanism == 'carbonation' else 1
    sl_fig.add_annotation(
        x=mechs[gov_idx], y=t_inits_plot[gov_idx] + t_props[gov_idx] + 10,
        text=f"⚑ Gouvernant<br>SL = {min(t_inits_plot[gov_idx]+t_props[gov_idx], 600):.0f} ans",
        font=dict(size=11, color='darkred'), showarrow=True,
        arrowcolor='darkred', arrowhead=2,
    )
    sl_fig.update_layout(
        barmode='stack',
        title='Durée de vie estimée par mécanisme de dégradation',
        xaxis_title='Mécanisme', yaxis_title='Durée de vie (ans)',
        template='plotly_white', legend=dict(x=0.75, y=0.95),
    )

    # ── Composition pie + bar
    labels = ['Ciment', 'Eau', 'Cendres vol.', 'Laitier',
              'Fumée silice', 'Sable', 'Gravier']
    values = [mix.cement, mix.water, mix.fly_ash, mix.slag,
              mix.silica_fume, mix.fine_agg, mix.coarse_agg]
    pie_colors = ['#e74c3c','#3498db','#95a5a6','#7f8c8d',
                   '#bdc3c7','#f39c12','#e67e22']

    comp_fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'pie'}, {'type': 'bar'}]],
        subplot_titles=('Composition massique', 'CO₂ par composant'),
    )
    comp_fig.add_trace(go.Pie(
        labels=labels, values=values, marker_colors=pie_colors,
        hovertemplate='%{label}: %{value:.0f} kg/m³ (%{percent})<extra></extra>',
        textinfo='label+percent',
    ), row=1, col=1)

    co2_vals = [
        mix.cement * CO2_FACTORS['cement'],
        mix.water  * CO2_FACTORS['water'],
        mix.fly_ash * CO2_FACTORS['fly_ash'],
        mix.slag    * CO2_FACTORS['slag'],
        mix.silica_fume * CO2_FACTORS['silica_fume'],
        mix.fine_agg   * CO2_FACTORS['fine_agg'],
        mix.coarse_agg * CO2_FACTORS['coarse_agg'],
    ]
    comp_fig.add_trace(go.Bar(
        x=labels, y=co2_vals, marker_color=pie_colors,
        text=[f'{v:.1f}' for v in co2_vals], textposition='outside',
    ), row=1, col=2)
    comp_fig.update_layout(
        title_text=f"Composition  —  {concrete_class(fck)}  |  w/b = {mix.w_b:.3f}  |  CO₂ = {co2:.0f} kg/m³",
        showlegend=False, template='plotly_white',
    )

    # ── Sensitivity: SL and CO2 vs w/b for several cover values
    wb_arr = np.linspace(0.32, 0.65, 45)
    cover_vals = [20, 30, 40, 50, 65]
    blues2 = ['#aed6f1','#5dade2','#2e86c1','#1a5276','#0a2342']

    sens_fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Durée de vie vs w/b', 'CO₂ vs w/b'),
    )
    for cov_v, col in zip(cover_vals, blues2):
        sl_v, co2_v = [], []
        for wbv in wb_arr:
            m2 = MixDesign(cement=350, water=350 * wbv,
                            fine_agg=750, coarse_agg=1050)
            r  = estimate_service_life(m2, cov_v, carb_class, cl_class, rh)
            sl_v.append(r.t_service_total)
            co2_v.append(embodied_co2(m2))
        sens_fig.add_trace(go.Scatter(
            x=wb_arr, y=sl_v, name=f'enrobage {cov_v}mm',
            line=dict(color=col, width=2),
        ), row=1, col=1)
        sens_fig.add_trace(go.Scatter(
            x=wb_arr, y=co2_v, name=f'enrobage {cov_v}mm',
            showlegend=False, line=dict(color=col, width=2),
        ), row=1, col=2)
    # Mark current w/b
    sens_fig.add_vline(x=wb, line_dash='dot', line_color='red',
                        annotation_text=f'w/b actuel = {wb:.2f}')
    sens_fig.update_xaxes(title_text='Rapport w/b')
    sens_fig.update_yaxes(title_text='Durée de vie (ans)', row=1, col=1)
    sens_fig.update_yaxes(title_text='CO₂ (kg/m³)',        row=1, col=2)
    sens_fig.update_layout(template='plotly_white', hovermode='x unified',
                            legend=dict(x=1.02, y=1))

    return kpis + [carb_fig, cl_fig, sl_fig, comp_fig, sens_fig]


# Expose WSGI server for Gunicorn (Render / Railway / Heroku)
server = app.server

if __name__ == '__main__':
    print("Dashboard disponible sur http://127.0.0.1:8050")
    app.run(debug=False, port=8050)
