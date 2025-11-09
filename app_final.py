"""
ForestED PROFESSIONAL - Application Dash Expert
================================================
Version professionnelle avec design shadcn/ui, synchronisation des visualisations,
filtres interactifs, et ergonomie optimale.
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output
import numpy as np
from datetime import datetime

# ========== CONFIGURATION DE L'APPLICATION ==========

app = dash.Dash(
    __name__, 
    suppress_callback_exceptions=True,
    assets_folder='assets',
    title="ForestED Professional"
)

# ========== CHARGEMENT DES DONNEES ==========

df = pd.read_excel("all_Competence.xlsx")

# Parser les competences
def parse_competencies(row):
    competencies = []
    for i in range(1, 13):
        comp_col = f"Competencies.{i}"
        if comp_col in row.index and pd.notna(row[comp_col]):
            competencies.append(row[comp_col])
    return competencies

df["Competencies_List"] = df.apply(parse_competencies, axis=1)

# Extraire UE
def extract_ue(ue_string):
    if pd.isna(ue_string):
        return None
    parts = [part.strip() for part in ue_string.split('-')]
    for part in parts:
        if part.startswith("UE"):
            return part
    return None

df['UE_Clean'] = df['UE'].apply(extract_ue)

# ========== CONFIGURATION DES COULEURS ==========

COLORS = {
    'primary': '#3B82F6',
    'secondary': '#8B5CF6',
    'success': '#10B981',
    'warning': '#F59E0B',
    'danger': '#EF4444',
    'info': '#06B6D4',
    'light': '#F8FAFC',
    'dark': '#1E293B',
    'tc': ['#3B82F6', '#8B5CF6', '#EC4899', '#F97316'],
    'idu': ['#10B981', '#14B8A6', '#06B6D4', '#0EA5E9'],
    'levels': {
        'N': '#BFDBFE',
        'A': '#60A5FA',
        'M': '#2563EB',
        'E': '#1E40AF',
    },
    'ue': {
        'UE1': '#E25012',
        'UE2': '#E28A12',
        'UE3': '#155992',
        'UE4': '#0C9A61',
    }
}

CATEGORY_COLORS = {
    "IDU-1": "#FF513F",
    "IDU-2": "#FFA03F",
    "IDU-3": "#DC3785",
    "IDU-4": "#FFC83F",
    "TC-1": "#2DB593",
    "TC-2": "#3A6CB7",
    "TC-3": "#65E038",
    "TC-4": "#5740BD",
}

def get_competency_color(competency):
    parts = competency.split("-")
    if len(parts) < 2:
        return "#777777"
    category_key = f"{parts[0]}-{parts[1][0]}"
    return CATEGORY_COLORS.get(category_key, "#AAAAAA")

# ========== COMPOSANTS UI REUTILISABLES ==========

def create_stat_card(title, value, subtitle, icon, color, trend=None):
    card_content = [
        html.Div([
            html.Div(title, className="stat-label"),
            html.Div(icon, className="stat-icon", style={'background': f'{color}20', 'color': color})
        ], className="stat-header"),
        html.Div(str(value), className="stat-value", style={'color': color}),
    ]
    
    if trend:
        trend_class = "stat-trend-up" if trend > 0 else "stat-trend-down"
        card_content.append(
            html.Div([
                html.Span("↑" if trend > 0 else "↓"),
                html.Span(f"{abs(trend)}% vs moyenne")
            ], className=f"stat-trend {trend_class}")
        )
    
    card_content.append(html.Div(subtitle, className="stat-description"))
    
    return html.Div(card_content, className="card stat-card card-content")

def create_filter_section():
    return html.Div([
        html.Div([
            html.H3("Filtres Interactifs", className="card-title"),
            html.P("Personnalisez votre vue en filtrant les donnees", className="card-description")
        ], className="card-header"),
        html.Div([
            html.Div([
                html.Div([
                    html.Label("Semestre", className="filter-label"),
                    dcc.Dropdown(
                        id='semester-filter',
                        options=[{'label': f'Semestre {s}', 'value': s} for s in sorted(df['Semestre'].unique())],
                        value=sorted(df['Semestre'].unique()),
                        multi=True,
                        className="filter-input",
                        placeholder="Selectionner les semestres..."
                    )
                ], className="filter-group"),
                
                html.Div([
                    html.Label("Unite d'Enseignement", className="filter-label"),
                    dcc.Dropdown(
                        id='ue-filter',
                        options=[{'label': ue, 'value': ue} for ue in sorted(df['UE_Clean'].dropna().unique())],
                        value=sorted(df['UE_Clean'].dropna().unique()),
                        multi=True,
                        className="filter-input",
                        placeholder="Selectionner les UE..."
                    )
                ], className="filter-group"),
                
                html.Div([
                    html.Label("Niveau de Maitrise", className="filter-label"),
                    html.Div([
                        html.Span([
                            dcc.Checklist(
                                id='level-filter',
                                options=[
                                    {'label': ' Novice', 'value': 'N'},
                                    {'label': ' Apprenti', 'value': 'A'},
                                    {'label': ' Maitre', 'value': 'M'},
                                    {'label': ' Expert', 'value': 'E'}
                                ],
                                value=['N', 'A', 'M', 'E'],
                                inline=True,
                                className="filter-badges"
                            )
                        ])
                    ])
                ], className="filter-group"),
            ], className="filter-grid"),
            
            html.Div([
                html.Div([
                    html.Span("", style={'marginRight': '8px'}),
                    html.Span(id='filter-status', children="Tous les elements affiches")
                ], className="filter-status"),
                html.Button(
                    "Reinitialiser les filtres",
                    id='reset-filters',
                    className="btn btn-outline btn-sm"
                )
            ], className="filter-footer")
        ], className="card-content")
    ], className="card filter-panel")

# ========== FONCTIONS DE VISUALISATION ==========

def create_personal_roadmap(filtered_df):
    semesters = sorted(filtered_df['Semestre'].unique())
    all_comps = filtered_df['Competencies_List'].explode().dropna().unique()
    
    progression_data = []
    for comp in all_comps:
        comp_type = comp.split('-')[0]
        for sem in semesters:
            count = filtered_df[filtered_df['Semestre'] == sem]['Competencies_List'].apply(
                lambda x: comp in x
            ).sum()
            
            levels_this_sem = filtered_df[
                (filtered_df['Semestre'] == sem) & 
                (filtered_df['Competencies_List'].apply(lambda x: comp in x))
            ]['Level']
            
            if count > 0 and not levels_this_sem.empty:
                level_order = {'N': 1, 'A': 2, 'M': 3, 'E': 4}
                avg_level = levels_this_sem.map(level_order).mean()
                
                progression_data.append({
                    'Competence': comp,
                    'Semestre': f'S{sem}',
                    'Type': comp_type,
                    'Intensite': count,
                    'Niveau': avg_level
                })
    
    prog_df = pd.DataFrame(progression_data)
    
    if prog_df.empty:
        return go.Figure().add_annotation(
            text="Aucune donnee disponible pour les filtres selectionnes",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color=COLORS['dark'])
        )
    
    pivot = prog_df.pivot_table(
        index='Competence',
        columns='Semestre',
        values='Niveau',
        aggfunc='mean'
    )
    
    fig = go.Figure()
    
    fig.add_trace(go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale=[
            [0, COLORS['levels']['N']],
            [0.33, COLORS['levels']['A']],
            [0.66, COLORS['levels']['M']],
            [1, COLORS['levels']['E']]
        ],
        text=pivot.values.round(1),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(
            title="Niveau",
            tickvals=[1, 2, 3, 4],
            ticktext=['Novice', 'Apprenti', 'Maitre', 'Expert']
        ),
        hovertemplate='<b>%{y}</b><br>%{x}<br>Niveau: %{z:.1f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Feuille de Route - Progression des Competences",
        xaxis_title="Semestre",
        yaxis_title="Competence",
        height=700,
        font=dict(size=11, family='Inter, sans-serif'),
        plot_bgcolor='white',
        paper_bgcolor=COLORS['light'],
        margin=dict(l=200, r=50, t=80, b=50)
    )
    
    return fig

def create_workload_prediction(filtered_df):
    semesters = sorted(filtered_df['Semestre'].unique())
    
    workload_data = []
    for sem in semesters:
        sem_data = filtered_df[filtered_df['Semestre'] == sem]
        
        unique_comps = set()
        for comp_list in sem_data['Competencies_List']:
            unique_comps.update(comp_list)
        
        level_counts = sem_data['Level'].value_counts()
        
        workload_data.append({
            'Semestre': f'S{sem}',
            'Competences_Uniques': len(unique_comps),
            'Novice': level_counts.get('N', 0),
            'Apprenti': level_counts.get('A', 0),
            'Maitre': level_counts.get('M', 0),
            'Expert': level_counts.get('E', 0),
        })
    
    wl_df = pd.DataFrame(workload_data)
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(go.Bar(
        name='Expert',
        x=wl_df['Semestre'],
        y=wl_df['Expert'],
        marker_color=COLORS['levels']['E'],
        hovertemplate='Expert: %{y}<extra></extra>'
    ), secondary_y=False)
    
    fig.add_trace(go.Bar(
        name='Maitre',
        x=wl_df['Semestre'],
        y=wl_df['Maitre'],
        marker_color=COLORS['levels']['M'],
        hovertemplate='Maitre: %{y}<extra></extra>'
    ), secondary_y=False)
    
    fig.add_trace(go.Bar(
        name='Apprenti',
        x=wl_df['Semestre'],
        y=wl_df['Apprenti'],
        marker_color=COLORS['levels']['A'],
        hovertemplate='Apprenti: %{y}<extra></extra>'
    ), secondary_y=False)
    
    fig.add_trace(go.Bar(
        name='Novice',
        x=wl_df['Semestre'],
        y=wl_df['Novice'],
        marker_color=COLORS['levels']['N'],
        hovertemplate='Novice: %{y}<extra></extra>'
    ), secondary_y=False)
    
    fig.add_trace(go.Scatter(
        name='Competences Uniques',
        x=wl_df['Semestre'],
        y=wl_df['Competences_Uniques'],
        mode='lines+markers',
        line=dict(color=COLORS['danger'], width=3),
        marker=dict(size=10),
        hovertemplate='Competences uniques: %{y}<extra></extra>'
    ), secondary_y=True)
    
    fig.update_layout(
        title="Charge de Travail par Semestre",
        xaxis_title="Semestre",
        barmode='stack',
        height=600,
        font=dict(size=11, family='Inter, sans-serif'),
        plot_bgcolor='white',
        paper_bgcolor=COLORS['light'],
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    fig.update_yaxes(title_text="Nombre d'objectifs", secondary_y=False)
    fig.update_yaxes(title_text="Competences uniques", secondary_y=True)
    
    return fig

def create_competency_radar(filtered_df):
    categories = ['TC-1', 'TC-2', 'TC-3', 'TC-4', 'IDU-1', 'IDU-2', 'IDU-3', 'IDU-4']
    
    counts = []
    for cat in categories:
        count = 0
        for comps in filtered_df['Competencies_List']:
            count += sum(1 for c in comps if c.startswith(cat))
        counts.append(count)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=counts,
        theta=categories,
        fill='toself',
        name='Couverture',
        line=dict(color=COLORS['primary'], width=2),
        fillcolor=f'rgba(59, 130, 246, 0.3)'
    ))
    
    fig.update_layout(
        title="Equilibre des Competences",
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(counts) * 1.1]
            )
        ),
        height=600,
        font=dict(size=11, family='Inter, sans-serif'),
        paper_bgcolor=COLORS['light']
    )
    
    return fig

def create_critical_competencies(filtered_df):
    comp_freq = {}
    comp_levels = {}
    
    for idx, row in filtered_df.iterrows():
        for comp in row['Competencies_List']:
            comp_freq[comp] = comp_freq.get(comp, 0) + 1
            if comp not in comp_levels:
                comp_levels[comp] = []
            comp_levels[comp].append(row['Level'])
    
    top_comps = sorted(comp_freq.items(), key=lambda x: x[1], reverse=True)[:15]
    
    fig = go.Figure()
    
    for comp, freq in top_comps:
        levels = comp_levels[comp]
        level_order = {'N': 1, 'A': 2, 'M': 3, 'E': 4}
        avg_level = np.mean([level_order[l] for l in levels if l in level_order])
        
        color = get_competency_color(comp)
        
        fig.add_trace(go.Bar(
            x=[freq],
            y=[comp],
            orientation='h',
            name=comp,
            marker_color=color,
            hovertemplate=f'<b>{comp}</b><br>Frequence: %{{x}}<br>Niveau moyen: {avg_level:.1f}<extra></extra>',
            showlegend=False
        ))
    
    fig.update_layout(
        title="Competences Critiques (Top 15)",
        xaxis_title="Frequence d'apparition",
        yaxis_title="Competence",
        height=600,
        font=dict(size=11, family='Inter, sans-serif'),
        plot_bgcolor='white',
        paper_bgcolor=COLORS['light']
    )
    
    return fig

def create_sunburst_hierarchy(filtered_df):
    hierarchy_data = []
    
    for _, row in filtered_df.iterrows():
        sem = f"S{row['Semestre']}"
        ue = row['UE_Clean'] if pd.notna(row['UE_Clean']) else 'Autre'
        module = row['EC'][:30]
        
        for comp in row['Competencies_List']:
            hierarchy_data.append({
                'Semestre': sem,
                'UE': ue,
                'Module': module,
                'Competence': comp,
                'Value': 1
            })
    
    hierarchy_df = pd.DataFrame(hierarchy_data)
    
    if hierarchy_df.empty:
        return go.Figure()
    
    fig = px.sunburst(
        hierarchy_df,
        path=['Semestre', 'UE', 'Module', 'Competence'],
        values='Value',
        color='Competence',
        color_discrete_map={comp: get_competency_color(comp) for comp in hierarchy_df['Competence'].unique()}
    )
    
    fig.update_layout(
        title="Vue Hierarchique du Curriculum",
        height=700,
        font=dict(size=11, family='Inter, sans-serif'),
        paper_bgcolor=COLORS['light']
    )
    
    return fig

def calculate_statistics(filtered_df):
    total_modules = filtered_df['EC'].nunique()
    total_competencies = len(set([comp for comps in filtered_df['Competencies_List'] for comp in comps]))
    total_objectives = len(filtered_df)
    avg_comp_per_module = total_competencies / total_modules if total_modules > 0 else 0
    
    return {
        'total_modules': total_modules,
        'total_competencies': total_competencies,
        'total_objectives': total_objectives,
        'avg_comp_per_module': round(avg_comp_per_module, 1)
    }

# ========== LAYOUT DE L'APPLICATION ==========

app.layout = html.Div([
    dcc.Store(id='filtered-data-store'),
    
    html.Div([
        html.Div([
            html.Div([
                html.Div("", className="logo-icon"),
                html.Div([
                    html.H1("ForestED Professional", className="header-title"),
                    html.P("Tableau de bord intelligent pour le suivi des competences", className="header-subtitle")
                ])
            ], className="header-logo"),
            html.Div([
                html.Span(f"Derniere mise a jour: {datetime.now().strftime('%d/%m/%Y')}", 
                         style={'fontSize': '14px', 'opacity': 0.9})
            ], className="header-actions")
        ], className="header-content")
    ], className="app-header"),
    
    html.Div([
        html.Div([
            html.H2("Vue d'Ensemble", className="font-bold mb-lg", style={'color': COLORS['dark']}),
            html.Div(id='stats-cards-container', className="stats-grid mb-xl")
        ], className="container mt-xl"),
        
        html.Div([
            create_filter_section()
        ], className="container mb-xl"),
        
        html.Div([
            html.Div([
                html.Div([
                    dcc.Graph(id='roadmap-chart', config={'displayModeBar': True, 'displaylogo': False})
                ], className="card-content")
            ], className="card mb-lg"),
            
            html.Div([
                html.Div([
                    html.Div([
                        dcc.Graph(id='workload-chart', config={'displayModeBar': True, 'displaylogo': False})
                    ], className="card-content")
                ], className="card", style={'marginBottom': '24px'}),
                
                html.Div([
                    html.Div([
                        dcc.Graph(id='radar-chart', config={'displayModeBar': True, 'displaylogo': False})
                    ], className="card-content")
                ], className="card")
            ], style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '24px'}, className="mb-lg"),
            
            html.Div([
                html.Div([
                    html.Div([
                        dcc.Graph(id='critical-chart', config={'displayModeBar': True, 'displaylogo': False})
                    ], className="card-content")
                ], className="card", style={'marginBottom': '24px'}),
                
                html.Div([
                    html.Div([
                        dcc.Graph(id='sunburst-chart', config={'displayModeBar': True, 'displaylogo': False})
                    ], className="card-content")
                ], className="card")
            ], style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '24px'}, className="mb-lg"),
        ], className="container"),
        
    ], className="app-container"),
    
    html.Div([
        html.Div([
            html.P("Astuce: Utilisez les filtres pour personnaliser votre vue", 
                  style={'textAlign': 'center', 'margin': '0', 'fontSize': '14px', 'color': COLORS['dark']}),
            html.P("ForestED Professional - 2025", 
                  style={'textAlign': 'center', 'margin': '10px 0 0 0', 'fontSize': '12px', 'color': '#64748b'})
        ])
    ], style={'padding': '40px 20px', 'backgroundColor': COLORS['light'], 'marginTop': '60px'})
    
], style={'fontFamily': 'Inter, sans-serif', 'backgroundColor': 'white', 'minHeight': '100vh'})

# ========== CALLBACKS ==========

@app.callback(
    [Output('filtered-data-store', 'data'),
     Output('filter-status', 'children')],
    [Input('semester-filter', 'value'),
     Input('ue-filter', 'value'),
     Input('level-filter', 'value'),
     Input('reset-filters', 'n_clicks')],
    prevent_initial_call=False
)
def filter_data(semesters, ues, levels, reset_clicks):
    ctx = dash.callback_context
    
    if ctx.triggered and ctx.triggered[0]['prop_id'] == 'reset-filters.n_clicks':
        filtered_df = df
        status = f"Tous les elements affiches ({len(df)} objectifs)"
    else:
        filtered_df = df.copy()
        
        if semesters:
            filtered_df = filtered_df[filtered_df['Semestre'].isin(semesters)]
        if ues:
            filtered_df = filtered_df[filtered_df['UE_Clean'].isin(ues)]
        if levels:
            filtered_df = filtered_df[filtered_df['Level'].isin(levels)]
        
        status = f"{len(filtered_df)} objectifs affiches sur {len(df)} total"
    
    return filtered_df.to_json(date_format='iso', orient='split'), status

@app.callback(
    Output('stats-cards-container', 'children'),
    Input('filtered-data-store', 'data')
)
def update_stats_cards(filtered_data_json):
    if filtered_data_json is None:
        filtered_df = df
    else:
        filtered_df = pd.read_json(filtered_data_json, orient='split')
        filtered_df["Competencies_List"] = filtered_df.apply(parse_competencies, axis=1)
    
    stats = calculate_statistics(filtered_df)
    
    return [
        create_stat_card(
            "Modules Totaux",
            stats['total_modules'],
            "Nombre de modules dans le curriculum",
            "",
            COLORS['primary']
        ),
        create_stat_card(
            "Competences",
            stats['total_competencies'],
            "Competences uniques developpees",
            "",
            COLORS['success'],
            trend=5
        ),
        create_stat_card(
            "Objectifs",
            stats['total_objectives'],
            "Objectifs d'apprentissage au total",
            "",
            COLORS['info']
        ),
        create_stat_card(
            "Moy. Comp/Module",
            stats['avg_comp_per_module'],
            "Competences moyennes par module",
            "",
            COLORS['warning']
        )
    ]

@app.callback(
    Output('roadmap-chart', 'figure'),
    Input('filtered-data-store', 'data')
)
def update_roadmap(filtered_data_json):
    if filtered_data_json is None:
        filtered_df = df
    else:
        filtered_df = pd.read_json(filtered_data_json, orient='split')
        filtered_df["Competencies_List"] = filtered_df.apply(parse_competencies, axis=1)
    
    return create_personal_roadmap(filtered_df)

@app.callback(
    Output('workload-chart', 'figure'),
    Input('filtered-data-store', 'data')
)
def update_workload(filtered_data_json):
    if filtered_data_json is None:
        filtered_df = df
    else:
        filtered_df = pd.read_json(filtered_data_json, orient='split')
        filtered_df["Competencies_List"] = filtered_df.apply(parse_competencies, axis=1)
    
    return create_workload_prediction(filtered_df)

@app.callback(
    Output('radar-chart', 'figure'),
    Input('filtered-data-store', 'data')
)
def update_radar(filtered_data_json):
    if filtered_data_json is None:
        filtered_df = df
    else:
        filtered_df = pd.read_json(filtered_data_json, orient='split')
        filtered_df["Competencies_List"] = filtered_df.apply(parse_competencies, axis=1)
    
    return create_competency_radar(filtered_df)

@app.callback(
    Output('critical-chart', 'figure'),
    Input('filtered-data-store', 'data')
)
def update_critical(filtered_data_json):
    if filtered_data_json is None:
        filtered_df = df
    else:
        filtered_df = pd.read_json(filtered_data_json, orient='split')
        filtered_df["Competencies_List"] = filtered_df.apply(parse_competencies, axis=1)
    
    return create_critical_competencies(filtered_df)

@app.callback(
    Output('sunburst-chart', 'figure'),
    Input('filtered-data-store', 'data')
)
def update_sunburst(filtered_data_json):
    if filtered_data_json is None:
        filtered_df = df
    else:
        filtered_df = pd.read_json(filtered_data_json, orient='split')
        filtered_df["Competencies_List"] = filtered_df.apply(parse_competencies, axis=1)
    
    return create_sunburst_hierarchy(filtered_df)

# ========== LANCEMENT DE L'APPLICATION ==========

if __name__ == '__main__':
    app.run_server(debug=False, host='0.0.0.0', port=8050)