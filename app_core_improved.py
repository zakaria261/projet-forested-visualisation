# app_core_improved.py — VERSION AMÉLIORÉE AVEC VISUALISATIONS INNOVANTES
# Focus: Perspective étudiant - comprendre son parcours et s'organiser efficacement

import random
import pandas as pd
import networkx as nx
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import dash
from dash import dcc, html
import colorsys
from copy import copy
import numpy as np

# Instance Dash
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Lecture des données Excel
file_path = "all_Competence.xlsx"
df = pd.read_excel(file_path)

# ==================== CONFIGURATION COULEURS ====================
UE_colors = {
    "UE1": "#E25012",
    "UE2": "#E28A12",
    "UE3": "#155992",
    "UE4": "#0C9A61",
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

def extract_ue(ue_string):
    if pd.isna(ue_string):
        return None
    ue_parts = [part.strip() for part in ue_string.split('-')]
    for part in ue_parts:
        if part.startswith("UE"):
            return part
    return None

df['UE'] = df['UE'].apply(extract_ue)

def get_ue_color(ue):
    return UE_colors.get(ue, "#7f7f7f")

def adjust_color_brightness(hex_color, factor):
    rgb = [int(hex_color[i:i+2], 16) for i in (1, 3, 5)]
    h, l, s = colorsys.rgb_to_hls(rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)
    l = max(0, min(1, l * factor))
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"

def get_level_color(level, ue_or_base_color):
    if level == 'N':
        factor = 1.5
    elif level == 'A':
        factor = 1.2
    elif level == 'M':
        factor = 0.8
    elif level == 'E':
        factor = 0.6
    else:
        factor = 1.0
    return adjust_color_brightness(ue_or_base_color, factor)

def get_competency_color(competency, level=None):
    parts = competency.split("-")
    if len(parts) < 2:
        return "#777777"
    category_key = f"{parts[0]}-{parts[1][0]}"
    base_color = CATEGORY_COLORS.get(category_key, "#AAAAAA")
    return get_level_color(level, base_color) if level else base_color

def get_category_base_color(category_key):
    return CATEGORY_COLORS.get(category_key, "#777777")

# ==================== PARSING & PRÉPARATION ====================
def parse_competencies(row):
    competencies = []
    for i in range(1, 13):
        comp_col = f"Competencies.{i}"
        if comp_col in row and pd.notna(row[comp_col]):
            competencies.append(row[comp_col])
    return competencies

def get_competency_type(comp):
    if str(comp).startswith('TC'):
        return 'TC'
    if str(comp).startswith('IDU'):
        return 'IDU'
    return None

df["All_Competencies"] = df.apply(parse_competencies, axis=1)
df['Competency_Type'] = df['All_Competencies'].apply(lambda lst: [get_competency_type(comp) for comp in lst])

# ==================== NOUVELLE VIZ 1: RADAR CHART PAR SEMESTRE ====================
def create_competency_radar_by_semester():
    """
    Graphique radar montrant la couverture des compétences par semestre
    INNOVATION: Permet à l'étudiant de voir instantanément quels domaines sont travaillés chaque semestre
    """
    semesters = sorted(df['Semestre'].dropna().unique())
    
    # Compter les compétences par catégorie et par semestre
    competency_categories = ['TC-1', 'TC-2', 'TC-3', 'TC-4', 'IDU-1', 'IDU-2', 'IDU-3', 'IDU-4']
    
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[f'Semestre {s}' for s in semesters],
        specs=[[{'type': 'polar'}]*3, [{'type': 'polar'}]*3]
    )
    
    positions = [(1,1), (1,2), (1,3), (2,1), (2,2), (2,3)]
    
    for idx, sem in enumerate(semesters):
        if idx >= len(positions):
            break
            
        sem_data = df[df['Semestre'] == sem]
        counts = []
        
        for cat in competency_categories:
            count = 0
            for comps in sem_data['All_Competencies']:
                count += sum(1 for c in comps if c.startswith(cat))
            counts.append(count)
        
        row, col = positions[idx]
        
        fig.add_trace(go.Scatterpolar(
            r=counts,
            theta=competency_categories,
            fill='toself',
            name=f'S{sem}',
            line=dict(color=UE_colors.get('UE1', '#E25012')),
            fillcolor='rgba(226, 80, 18, 0.3)'
        ), row=row, col=col)
    
    fig.update_layout(
        height=800,
        title_text="🎯 Radar de Compétences par Semestre<br><sub>Vue d'ensemble de la couverture des domaines de compétences</sub>",
        showlegend=False
    )
    
    return fig

# ==================== NOUVELLE VIZ 2: SANKEY DIAGRAM ====================
def create_competency_flow_sankey():
    """
    Diagramme de Sankey montrant le flux des compétences à travers les semestres
    INNOVATION: Visualise comment les compétences se développent et s'interconnectent dans le temps
    """
    semesters = sorted(df['Semestre'].dropna().unique())
    
    # Créer les liens entre semestres pour chaque compétence
    source = []
    target = []
    value = []
    labels = []
    colors = []
    
    # Créer les labels pour chaque semestre
    for sem in semesters:
        labels.append(f"S{sem}")
    
    # Ajouter les compétences comme nœuds intermédiaires
    all_comps = set()
    for comps in df['All_Competencies']:
        all_comps.update(comps)
    
    comp_to_idx = {comp: idx + len(semesters) for idx, comp in enumerate(sorted(all_comps))}
    labels.extend(sorted(all_comps))
    
    # Créer les liens
    for sem_idx, sem in enumerate(semesters):
        sem_data = df[df['Semestre'] == sem]
        comp_counts = {}
        
        for comps in sem_data['All_Competencies']:
            for comp in comps:
                comp_counts[comp] = comp_counts.get(comp, 0) + 1
        
        for comp, count in comp_counts.items():
            source.append(sem_idx)
            target.append(comp_to_idx[comp])
            value.append(count)
            colors.append(get_competency_color(comp))
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            color=['#E25012' if l.startswith('S') else get_competency_color(l) for l in labels]
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            color=['rgba(0,0,0,0.2)'] * len(source)
        )
    )])
    
    fig.update_layout(
        title_text="🌊 Flux des Compétences à Travers les Semestres<br><sub>Visualisation du développement progressif des compétences</sub>",
        font_size=10,
        height=600
    )
    
    return fig

# ==================== NOUVELLE VIZ 3: TIMELINE DE CHARGE DE TRAVAIL ====================
def create_workload_heatmap():
    """
    Carte de chaleur montrant la charge de travail par semestre et par UE
    INNOVATION: Aide l'étudiant à anticiper les périodes intenses et à mieux s'organiser
    """
    # Compter le nombre d'objectifs (Goals) par semestre et par UE
    workload_data = df.groupby(['Semestre', 'UE']).size().reset_index(name='Charge')
    
    # Créer une matrice pivot
    pivot_table = workload_data.pivot(index='UE', columns='Semestre', values='Charge').fillna(0)
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot_table.values,
        x=[f'Semestre {s}' for s in pivot_table.columns],
        y=pivot_table.index,
        colorscale='RdYlGn_r',  # Rouge = beaucoup, Vert = peu
        text=pivot_table.values,
        texttemplate='%{text}',
        textfont={"size": 12},
        colorbar=dict(title="Nombre<br>d'objectifs")
    ))
    
    fig.update_layout(
        title_text="📊 Carte de Charge de Travail par Semestre<br><sub>Anticipez les périodes intenses pour mieux vous organiser</sub>",
        xaxis_title="Semestre",
        yaxis_title="Unité d'Enseignement",
        height=500
    )
    
    return fig

# ==================== NOUVELLE VIZ 4: PROGRESSION CHRONOLOGIQUE ====================
def create_competency_progression_timeline():
    """
    Timeline interactive montrant l'évolution du nombre de compétences uniques acquises
    INNOVATION: Montre la progression cumulative des compétences pour motiver l'étudiant
    """
    semesters = sorted(df['Semestre'].dropna().unique())
    
    # Compter les compétences cumulatives
    tc_cumulative = []
    idu_cumulative = []
    all_tc = set()
    all_idu = set()
    
    for sem in semesters:
        sem_data = df[df['Semestre'] <= sem]
        
        for comps in sem_data['All_Competencies']:
            for comp in comps:
                if comp.startswith('TC'):
                    all_tc.add(comp)
                elif comp.startswith('IDU'):
                    all_idu.add(comp)
        
        tc_cumulative.append(len(all_tc))
        idu_cumulative.append(len(all_idu))
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=semesters,
        y=tc_cumulative,
        mode='lines+markers',
        name='Compétences TC',
        line=dict(color='#3A6CB7', width=3),
        marker=dict(size=10),
        fill='tonexty'
    ))
    
    fig.add_trace(go.Scatter(
        x=semesters,
        y=idu_cumulative,
        mode='lines+markers',
        name='Compétences IDU',
        line=dict(color='#FF513F', width=3),
        marker=dict(size=10),
        fill='tozeroy'
    ))
    
    fig.update_layout(
        title_text="📈 Progression Cumulative des Compétences<br><sub>Suivez votre développement tout au long du cursus</sub>",
        xaxis_title="Semestre",
        yaxis_title="Nombre de compétences uniques",
        height=400,
        hovermode='x unified'
    )
    
    return fig

# ==================== NOUVELLE VIZ 5: MATRICE DE COMPÉTENCES ====================
def create_competency_module_matrix():
    """
    Matrice interactive montrant quels modules développent quelles compétences
    INNOVATION: Permet de voir rapidement où chaque compétence est travaillée
    """
    # Créer une liste de toutes les compétences uniques
    all_comps = set()
    for comps in df['All_Competencies']:
        all_comps.update(comps)
    all_comps = sorted(all_comps)
    
    # Créer une matrice modules x compétences
    modules = df['EC'].unique()[:30]  # Limiter pour la lisibilité
    
    matrix_data = []
    for module in modules:
        row = []
        module_data = df[df['EC'] == module]
        module_comps = set()
        for comps in module_data['All_Competencies']:
            module_comps.update(comps)
        
        for comp in all_comps:
            row.append(1 if comp in module_comps else 0)
        matrix_data.append(row)
    
    fig = go.Figure(data=go.Heatmap(
        z=matrix_data,
        x=all_comps,
        y=[m[:30] for m in modules],  # Tronquer les noms longs
        colorscale=[[0, '#f0f0f0'], [1, '#2DB593']],
        showscale=False,
        hoverongaps=False
    ))
    
    fig.update_layout(
        title_text="🎓 Matrice Modules × Compétences<br><sub>Découvrez où chaque compétence est développée</sub>",
        xaxis_title="Compétences",
        yaxis_title="Modules (30 premiers)",
        height=800,
        xaxis={'side': 'top'}
    )
    
    return fig

# ==================== NOUVELLE VIZ 6: SUNBURST HIÉRARCHIQUE ====================
def create_competency_sunburst():
    """
    Diagramme sunburst montrant la hiérarchie Semestre > UE > Compétences
    INNOVATION: Vue hiérarchique intuitive de l'organisation du curriculum
    """
    # Préparer les données hiérarchiques
    hierarchy_data = []
    
    for _, row in df.iterrows():
        sem = f"S{row['Semestre']}"
        ue = row['UE'] if pd.notna(row['UE']) else 'Autre'
        
        for comp in row['All_Competencies']:
            hierarchy_data.append({
                'Semestre': sem,
                'UE': ue,
                'Competence': comp,
                'Value': 1
            })
    
    hierarchy_df = pd.DataFrame(hierarchy_data)
    
    fig = px.sunburst(
        hierarchy_df,
        path=['Semestre', 'UE', 'Competence'],
        values='Value',
        color='Competence',
        color_discrete_map={comp: get_competency_color(comp) for comp in hierarchy_df['Competence'].unique()}
    )
    
    fig.update_layout(
        title_text="☀️ Vue Hiérarchique du Curriculum<br><sub>Explorez l'organisation Semestre → UE → Compétences</sub>",
        height=700
    )
    
    return fig

# ==================== NOUVELLE VIZ 7: STATISTIQUES PAR COMPÉTENCE ====================
def create_competency_stats_cards():
    """
    Cartes statistiques pour chaque grande catégorie de compétences
    INNOVATION: Vue d'ensemble quantitative pour comprendre la répartition
    """
    categories = ['TC-1', 'TC-2', 'TC-3', 'TC-4', 'IDU-1', 'IDU-2', 'IDU-3', 'IDU-4']
    
    stats_html = []
    
    for cat in categories:
        # Compter les occurrences
        count = 0
        modules_count = 0
        semesters = set()
        
        for idx, row in df.iterrows():
            has_cat = any(comp.startswith(cat) for comp in row['All_Competencies'])
            if has_cat:
                modules_count += 1
                semesters.add(row['Semestre'])
                count += sum(1 for comp in row['All_Competencies'] if comp.startswith(cat))
        
        card = html.Div([
            html.Div(cat, style={
                'fontSize': '24px',
                'fontWeight': 'bold',
                'color': get_category_base_color(cat),
                'marginBottom': '10px'
            }),
            html.Div([
                html.Div([
                    html.Span(str(count), style={'fontSize': '32px', 'fontWeight': 'bold'}),
                    html.Br(),
                    html.Span('occurrences', style={'fontSize': '12px', 'color': '#666'})
                ], style={'textAlign': 'center', 'flex': 1}),
                html.Div([
                    html.Span(str(modules_count), style={'fontSize': '32px', 'fontWeight': 'bold'}),
                    html.Br(),
                    html.Span('modules', style={'fontSize': '12px', 'color': '#666'})
                ], style={'textAlign': 'center', 'flex': 1}),
                html.Div([
                    html.Span(str(len(semesters)), style={'fontSize': '32px', 'fontWeight': 'bold'}),
                    html.Br(),
                    html.Span('semestres', style={'fontSize': '12px', 'color': '#666'})
                ], style={'textAlign': 'center', 'flex': 1}),
            ], style={'display': 'flex', 'justifyContent': 'space-around'})
        ], style={
            'border': f'2px solid {get_category_base_color(cat)}',
            'borderRadius': '10px',
            'padding': '20px',
            'margin': '10px',
            'backgroundColor': '#f9f9f9',
            'width': '200px'
        })
        
        stats_html.append(card)
    
    return html.Div(stats_html, style={
        'display': 'flex',
        'flexWrap': 'wrap',
        'justifyContent': 'center',
        'gap': '20px',
        'padding': '20px'
    })

# ==================== LAYOUT AMÉLIORÉ ====================
app.layout = html.Div([
    html.Div([
        html.H1("🎓 ForestED - Dashboard Étudiant Amélioré", style={
            'textAlign': 'center',
            'color': '#2c3e50',
            'marginBottom': '10px'
        }),
        html.P("Visualisez et comprenez votre parcours académique de manière intuitive", style={
            'textAlign': 'center',
            'color': '#7f8c8d',
            'fontSize': '18px'
        })
    ], style={'backgroundColor': '#ecf0f1', 'padding': '20px', 'marginBottom': '30px'}),
    
    # Section 1: Statistiques Globales
    html.Div([
        html.H2("📊 Statistiques par Catégorie de Compétences", style={
            'textAlign': 'center',
            'color': '#34495e',
            'marginBottom': '20px'
        }),
        html.Div(id='stats-cards-container')
    ], style={'marginBottom': '40px'}),
    
    # Section 2: Progression et Timeline
    html.Div([
        html.H2("📈 Votre Progression dans le Cursus", style={
            'textAlign': 'center',
            'color': '#34495e',
            'marginBottom': '20px'
        }),
        dcc.Graph(id='progression-timeline')
    ], style={'marginBottom': '40px'}),
    
    # Section 3: Radar Charts
    html.Div([
        html.H2("🎯 Couverture des Compétences par Semestre", style={
            'textAlign': 'center',
            'color': '#34495e',
            'marginBottom': '20px'
        }),
        dcc.Graph(id='radar-chart')
    ], style={'marginBottom': '40px'}),
    
    # Section 4: Charge de travail
    html.Div([
        html.H2("⚖️ Anticipez Votre Charge de Travail", style={
            'textAlign': 'center',
            'color': '#34495e',
            'marginBottom': '20px'
        }),
        dcc.Graph(id='workload-heatmap')
    ], style={'marginBottom': '40px'}),
    
    # Section 5: Sankey
    html.Div([
        html.H2("🌊 Flux des Compétences", style={
            'textAlign': 'center',
            'color': '#34495e',
            'marginBottom': '20px'
        }),
        dcc.Graph(id='sankey-diagram')
    ], style={'marginBottom': '40px'}),
    
    # Section 6: Sunburst
    html.Div([
        html.H2("☀️ Vue Hiérarchique du Curriculum", style={
            'textAlign': 'center',
            'color': '#34495e',
            'marginBottom': '20px'
        }),
        dcc.Graph(id='sunburst-chart')
    ], style={'marginBottom': '40px'}),
    
    # Section 7: Matrice
    html.Div([
        html.H2("🎓 Matrice Modules × Compétences", style={
            'textAlign': 'center',
            'color': '#34495e',
            'marginBottom': '20px'
        }),
        dcc.Graph(id='matrix-chart')
    ], style={'marginBottom': '40px'}),
])

# Export
__all__ = [
    'app',
    'create_competency_radar_by_semester',
    'create_competency_flow_sankey',
    'create_workload_heatmap',
    'create_competency_progression_timeline',
    'create_competency_module_matrix',
    'create_competency_sunburst',
    'create_competency_stats_cards'
]