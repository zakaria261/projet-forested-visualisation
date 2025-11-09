"""
ForestED Amélioré - Visualisation du curriculum centré ÉTUDIANT
================================================================
Cette version repense complètement les visualisations pour répondre aux besoins des étudiants:
- Parcours personnalisé de progression
- Prédiction de charge de travail
- Identification des compétences critiques
- Tableau de bord de réussite
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State
import numpy as np
from datetime import datetime

# ========== CHARGEMENT ET PRÉPARATION DES DONNÉES ==========

app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "ForestED - Vue Étudiant"

# Charger les données
df = pd.read_excel("all_Competence.xlsx")

# Parser les compétences
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

# Couleurs professionnelles et accessibles
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'success': '#06A77D',
    'warning': '#F77F00',
    'danger': '#D62828',
    'info': '#4EA8DE',
    'light': '#F4F4F9',
    'dark': '#2B2D42',
    'tc': ['#3A86FF', '#8338EC', '#FF006E', '#FB5607'],
    'idu': ['#06FFA5', '#10F1DA', '#00BBF9', '#0096C7'],
    'levels': {
        'N': '#90E0EF',  # Novice - bleu très clair
        'A': '#00B4D8',  # Apprenti - bleu clair
        'M': '#0077B6',  # Maître - bleu moyen
        'E': '#03045E',  # Expert - bleu foncé
    }
}

# ========== NOUVELLES VISUALISATIONS RÉVOLUTIONNAIRES ==========

def create_personal_roadmap():
    """
    VISUALISATION 1: Feuille de route personnelle
    Montre la progression des compétences semestre par semestre
    OBJECTIF ÉTUDIANT: "Quand vais-je apprendre quoi?"
    """
    
    # Calculer la progression des compétences par semestre
    semesters = sorted(df['Semestre'].unique())
    all_comps = df['Competencies_List'].explode().dropna().unique()
    
    # Matrice de progression
    progression_data = []
    for comp in all_comps:
        comp_type = comp.split('-')[0]
        for sem in semesters:
            # Compter les occurrences de cette compétence ce semestre
            count = df[df['Semestre'] == sem]['Competencies_List'].apply(lambda x: comp in x).sum()
            
            # Déterminer le niveau moyen pour cette compétence ce semestre
            levels_this_sem = df[(df['Semestre'] == sem) & 
                                (df['Competencies_List'].apply(lambda x: comp in x))]['Level']
            
            if count > 0 and not levels_this_sem.empty:
                level_order = {'N': 1, 'A': 2, 'M': 3, 'E': 4}
                avg_level = levels_this_sem.map(level_order).mean()
                
                progression_data.append({
                    'Competence': comp,
                    'Semestre': f'S{sem}',
                    'Type': comp_type,
                    'Intensite': count,
                    'Niveau': avg_level,
                    'Tooltip': f"{comp}<br>S{sem}: {count} module(s)<br>Niveau moyen: {avg_level:.1f}"
                })
    
    prog_df = pd.DataFrame(progression_data)
    
    if prog_df.empty:
        return go.Figure()
    
    # Créer une heatmap interactive
    pivot = prog_df.pivot_table(
        index='Competence',
        columns='Semestre',
        values='Niveau',
        aggfunc='mean'
    )
    
    fig = go.Figure()
    
    # Ajouter la heatmap
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
            ticktext=['Novice', 'Apprenti', 'Maître', 'Expert']
        ),
        hovertemplate='<b>%{y}</b><br>%{x}<br>Niveau: %{z:.1f}<extra></extra>'
    ))
    
    fig.update_layout(
        title={
            'text': "📚 VOTRE FEUILLE DE ROUTE - Progression des Compétences par Semestre",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': COLORS['dark']}
        },
        xaxis_title="Semestre",
        yaxis_title="Compétence",
        height=800,
        font=dict(size=12),
        plot_bgcolor='white',
        paper_bgcolor=COLORS['light']
    )
    
    return fig


def create_workload_prediction():
    """
    VISUALISATION 2: Prédicteur de charge de travail
    Montre combien de compétences à maîtriser par semestre
    OBJECTIF ÉTUDIANT: "Quel semestre sera le plus chargé?"
    """
    
    semesters = sorted(df['Semestre'].unique())
    
    workload_data = []
    for sem in semesters:
        sem_data = df[df['Semestre'] == sem]
        
        # Compter les compétences uniques
        unique_comps = set()
        for comp_list in sem_data['Competencies_List']:
            unique_comps.update(comp_list)
        
        # Compter par niveau
        level_counts = sem_data['Level'].value_counts()
        
        # Compter les modules
        modules = sem_data['EC'].nunique()
        
        workload_data.append({
            'Semestre': f'S{sem}',
            'Sem_Num': sem,
            'Competences_Uniques': len(unique_comps),
            'Modules': modules,
            'Novice': level_counts.get('N', 0),
            'Apprenti': level_counts.get('A', 0),
            'Maitre': level_counts.get('M', 0),
            'Expert': level_counts.get('E', 0),
            'Total_Objectifs': len(sem_data)
        })
    
    wl_df = pd.DataFrame(workload_data)
    
    # Créer un graphique en colonnes empilées
    fig = go.Figure()
    
    # Ajouter les barres par niveau
    fig.add_trace(go.Bar(
        name='Expert (E)',
        x=wl_df['Semestre'],
        y=wl_df['Expert'],
        marker_color=COLORS['levels']['E'],
        text=wl_df['Expert'],
        textposition='inside',
        hovertemplate='Expert: %{y}<extra></extra>'
    ))
    
    fig.add_trace(go.Bar(
        name='Maître (M)',
        x=wl_df['Semestre'],
        y=wl_df['Maitre'],
        marker_color=COLORS['levels']['M'],
        text=wl_df['Maitre'],
        textposition='inside',
        hovertemplate='Maître: %{y}<extra></extra>'
    ))
    
    fig.add_trace(go.Bar(
        name='Apprenti (A)',
        x=wl_df['Semestre'],
        y=wl_df['Apprenti'],
        marker_color=COLORS['levels']['A'],
        text=wl_df['Apprenti'],
        textposition='inside',
        hovertemplate='Apprenti: %{y}<extra></extra>'
    ))
    
    fig.add_trace(go.Bar(
        name='Novice (N)',
        x=wl_df['Semestre'],
        y=wl_df['Novice'],
        marker_color=COLORS['levels']['N'],
        text=wl_df['Novice'],
        textposition='inside',
        hovertemplate='Novice: %{y}<extra></extra>'
    ))
    
    # Ajouter une ligne pour les compétences uniques
    fig.add_trace(go.Scatter(
        name='Compétences uniques',
        x=wl_df['Semestre'],
        y=wl_df['Competences_Uniques'],
        mode='lines+markers+text',
        line=dict(color=COLORS['danger'], width=3),
        marker=dict(size=12, symbol='diamond'),
        text=wl_df['Competences_Uniques'],
        textposition='top center',
        yaxis='y2',
        hovertemplate='Compétences uniques: %{y}<extra></extra>'
    ))
    
    fig.update_layout(
        title={
            'text': "⚡ CHARGE DE TRAVAIL PRÉDITE - Objectifs par Semestre et Niveau",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': COLORS['dark']}
        },
        barmode='stack',
        xaxis_title="Semestre",
        yaxis_title="Nombre d'objectifs d'apprentissage",
        yaxis2=dict(
            title="Compétences uniques",
            overlaying='y',
            side='right',
            showgrid=False
        ),
        height=600,
        plot_bgcolor='white',
        paper_bgcolor=COLORS['light'],
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode='x unified'
    )
    
    return fig


def create_competency_radar():
    """
    VISUALISATION 3: Radar des compétences par domaine
    Montre l'équilibre entre TC et IDU à travers le curriculum
    OBJECTIF ÉTUDIANT: "Suis-je équilibré dans mes compétences?"
    """
    
    # Calculer les statistiques par catégorie
    categories = []
    values_total = []
    values_expert = []
    
    for prefix in ['TC-1', 'TC-2', 'TC-3', 'TC-4', 'IDU-1', 'IDU-2', 'IDU-3', 'IDU-4']:
        # Compter toutes les occurrences
        total = 0
        expert = 0
        
        for comp_list in df['Competencies_List']:
            for comp in comp_list:
                if comp.startswith(prefix):
                    total += 1
                    # Vérifier si c'est niveau Expert
                    idx = df[df['Competencies_List'].apply(lambda x: comp in x)].index
                    if not idx.empty:
                        levels = df.loc[idx, 'Level']
                        if (levels == 'E').any():
                            expert += 1
        
        if total > 0:
            categories.append(prefix)
            values_total.append(total)
            values_expert.append(expert)
    
    fig = go.Figure()
    
    # Ajouter le radar pour toutes les occurrences
    fig.add_trace(go.Scatterpolar(
        r=values_total,
        theta=categories,
        fill='toself',
        name='Total occurrences',
        line_color=COLORS['info'],
        fillcolor='rgba(78, 168, 222, 0.3)'
    ))
    
    # Ajouter le radar pour le niveau Expert
    fig.add_trace(go.Scatterpolar(
        r=values_expert,
        theta=categories,
        fill='toself',
        name='Niveau Expert',
        line_color=COLORS['danger'],
        fillcolor='rgba(214, 40, 40, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(values_total) * 1.1]
            )
        ),
        title={
            'text': "🎯 ÉQUILIBRE DES COMPÉTENCES - Répartition par Domaine",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': COLORS['dark']}
        },
        height=600,
        showlegend=True,
        paper_bgcolor=COLORS['light']
    )
    
    return fig


def create_critical_path():
    """
    VISUALISATION 4: Chemin critique des compétences
    Identifie les compétences qui apparaissent le plus souvent (donc critiques pour la réussite)
    OBJECTIF ÉTUDIANT: "Quelles compétences sont les plus importantes?"
    """
    
    # Compter les occurrences de chaque compétence
    comp_counts = {}
    comp_levels = {}
    
    for idx, row in df.iterrows():
        for comp in row['Competencies_List']:
            if comp not in comp_counts:
                comp_counts[comp] = 0
                comp_levels[comp] = {'N': 0, 'A': 0, 'M': 0, 'E': 0}
            
            comp_counts[comp] += 1
            level = row['Level']
            if level in comp_levels[comp]:
                comp_levels[comp][level] += 1
    
    # Trier par importance
    sorted_comps = sorted(comp_counts.items(), key=lambda x: x[1], reverse=True)[:15]
    
    # Préparer les données
    comps = [x[0] for x in sorted_comps]
    counts = [x[1] for x in sorted_comps]
    
    # Calculer le "score de criticité" (plus de occurrences + niveau élevé = plus critique)
    criticality = []
    for comp in comps:
        levels = comp_levels[comp]
        score = (levels['N'] * 1 + levels['A'] * 2 + levels['M'] * 3 + levels['E'] * 4) / sum(levels.values())
        criticality.append(score)
    
    fig = go.Figure()
    
    # Créer un graphique en barres avec gradient de couleur selon criticité
    fig.add_trace(go.Bar(
        y=comps,
        x=counts,
        orientation='h',
        marker=dict(
            color=criticality,
            colorscale='RdYlGn_r',  # Rouge = très critique, Vert = moins critique
            showscale=True,
            colorbar=dict(
                title="Niveau moyen",
                tickvals=[1, 2, 3, 4],
                ticktext=['Novice', 'Apprenti', 'Maître', 'Expert']
            )
        ),
        text=counts,
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Occurrences: %{x}<br>Niveau moyen: %{marker.color:.1f}<extra></extra>'
    ))
    
    fig.update_layout(
        title={
            'text': "🔥 COMPÉTENCES CRITIQUES - Top 15 des plus fréquentes (À MAÎTRISER ABSOLUMENT)",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': COLORS['dark']}
        },
        xaxis_title="Nombre d'occurrences dans le curriculum",
        yaxis_title="Compétence",
        height=700,
        plot_bgcolor='white',
        paper_bgcolor=COLORS['light'],
        font=dict(size=12)
    )
    
    return fig


def create_module_competency_matrix():
    """
    VISUALISATION 5: Matrice Module-Compétence Interactive
    Montre quels modules développent quelles compétences
    OBJECTIF ÉTUDIANT: "Ce module va m'apprendre quoi exactement?"
    """
    
    # Créer une matrice module x compétence
    modules = df['EC'].unique()[:20]  # Limiter pour la lisibilité
    all_comps = sorted(set([comp for comp_list in df['Competencies_List'] for comp in comp_list]))[:20]
    
    matrix_data = []
    
    for module in modules:
        module_data = df[df['EC'] == module]
        for comp in all_comps:
            # Compter combien de fois cette compétence apparaît dans ce module
            count = module_data['Competencies_List'].apply(lambda x: comp in x).sum()
            
            if count > 0:
                # Obtenir le semestre
                sem = module_data['Semestre'].iloc[0]
                matrix_data.append({
                    'Module': module,
                    'Competence': comp,
                    'Occurrences': count,
                    'Semestre': sem
                })
    
    matrix_df = pd.DataFrame(matrix_data)
    
    if matrix_df.empty:
        return go.Figure()
    
    # Créer la heatmap
    pivot = matrix_df.pivot_table(
        index='Module',
        columns='Competence',
        values='Occurrences',
        fill_value=0
    )
    
    fig = go.Figure()
    
    fig.add_trace(go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale='Viridis',
        text=pivot.values,
        texttemplate='%{text}',
        textfont={"size": 8},
        colorbar=dict(title="Occurrences"),
        hovertemplate='<b>%{y}</b><br>%{x}<br>Occurrences: %{z}<extra></extra>'
    ))
    
    fig.update_layout(
        title={
            'text': "🗺️ CARTE MODULE-COMPÉTENCE - Qui enseigne quoi?",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': COLORS['dark']}
        },
        xaxis_title="Compétence",
        yaxis_title="Module",
        height=800,
        font=dict(size=10),
        plot_bgcolor='white',
        paper_bgcolor=COLORS['light']
    )
    
    return fig


def create_semester_comparison_sunburst():
    """
    VISUALISATION 6: Comparaison Semestre en Sunburst
    Vue hiérarchique: Semestre > UE > Modules > Compétences
    OBJECTIF ÉTUDIANT: "Comment est organisé mon semestre?"
    """
    
    # Préparer les données hiérarchiques
    sunburst_data = []
    
    for _, row in df.iterrows():
        sem = f"S{row['Semestre']}"
        ue = row['UE_Clean'] if pd.notna(row['UE_Clean']) else 'UE?'
        module = row['EC']
        
        for comp in row['Competencies_List']:
            sunburst_data.append({
                'Semestre': sem,
                'UE': ue,
                'Module': module,
                'Competence': comp,
                'Level': row['Level']
            })
    
    sb_df = pd.DataFrame(sunburst_data)
    
    # Créer des labels hiérarchiques
    sb_df['labels'] = sb_df['Competence']
    sb_df['parents'] = sb_df['Module']
    
    # Ajouter les modules
    modules_df = sb_df[['Module', 'UE']].drop_duplicates()
    modules_df['labels'] = modules_df['Module']
    modules_df['parents'] = modules_df['UE']
    
    # Ajouter les UE
    ue_df = sb_df[['UE', 'Semestre']].drop_duplicates()
    ue_df['labels'] = ue_df['UE']
    ue_df['parents'] = ue_df['Semestre']
    
    # Ajouter les semestres
    sem_df = pd.DataFrame({'labels': sb_df['Semestre'].unique(), 'parents': ''})
    
    # Combiner
    hierarchy = pd.concat([
        sem_df[['labels', 'parents']],
        ue_df[['labels', 'parents']],
        modules_df[['labels', 'parents']],
        sb_df[['labels', 'parents']]
    ], ignore_index=True)
    
    # Limiter pour performance
    hierarchy = hierarchy.head(500)
    
    fig = go.Figure(go.Sunburst(
        labels=hierarchy['labels'],
        parents=hierarchy['parents'],
        marker=dict(
            colorscale='Rainbow',
            cmid=0
        ),
        hovertemplate='<b>%{label}</b><extra></extra>'
    ))
    
    fig.update_layout(
        title={
            'text': "🌞 VUE HIÉRARCHIQUE - Organisation du Curriculum",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': COLORS['dark']}
        },
        height=800,
        paper_bgcolor=COLORS['light']
    )
    
    return fig


# ========== LAYOUT DE L'APPLICATION ==========

app.layout = html.Div([
    # En-tête stylisé
    html.Div([
        html.H1("🎓 ForestED - Tableau de Bord ÉTUDIANT", 
                style={'textAlign': 'center', 'color': 'white', 'margin': '0', 'padding': '20px'}),
        html.P("Visualisez votre parcours d'apprentissage et planifiez votre réussite",
               style={'textAlign': 'center', 'color': 'white', 'margin': '0', 'fontSize': '16px'})
    ], style={
        'backgroundColor': COLORS['primary'],
        'marginBottom': '30px',
        'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'
    }),
    
    # Onglets pour naviguer entre les visualisations
    dcc.Tabs(id='tabs', value='tab-roadmap', children=[
        dcc.Tab(label='📚 Ma Feuille de Route', value='tab-roadmap',
                style={'fontWeight': 'bold', 'fontSize': '14px'}),
        dcc.Tab(label='⚡ Charge de Travail', value='tab-workload',
                style={'fontWeight': 'bold', 'fontSize': '14px'}),
        dcc.Tab(label='🎯 Équilibre Compétences', value='tab-radar',
                style={'fontWeight': 'bold', 'fontSize': '14px'}),
        dcc.Tab(label='🔥 Compétences Critiques', value='tab-critical',
                style={'fontWeight': 'bold', 'fontSize': '14px'}),
        dcc.Tab(label='🗺️ Carte Module-Compétence', value='tab-matrix',
                style={'fontWeight': 'bold', 'fontSize': '14px'}),
        dcc.Tab(label='🌞 Vue Hiérarchique', value='tab-sunburst',
                style={'fontWeight': 'bold', 'fontSize': '14px'}),
    ], style={'marginBottom': '20px'}),
    
    # Conteneur pour les graphiques
    html.Div(id='tabs-content', style={'padding': '20px'}),
    
    # Pied de page
    html.Div([
        html.P("💡 Astuce: Survolez les graphiques pour plus de détails. Cliquez pour zoomer.",
               style={'textAlign': 'center', 'color': COLORS['dark'], 'fontStyle': 'italic'})
    ], style={'marginTop': '50px', 'padding': '20px', 'backgroundColor': COLORS['light']})
    
], style={
    'fontFamily': 'Arial, sans-serif',
    'backgroundColor': 'white',
    'minHeight': '100vh'
})


# ========== CALLBACKS ==========

@app.callback(
    Output('tabs-content', 'children'),
    Input('tabs', 'value')
)
def render_content(tab):
    """Afficher le contenu selon l'onglet sélectionné"""
    
    if tab == 'tab-roadmap':
        return html.Div([
            html.H2("Votre Feuille de Route Personnelle", 
                    style={'color': COLORS['primary'], 'marginBottom': '20px'}),
            html.P([
                "Cette visualisation montre ",
                html.Strong("comment vos compétences progressent"),
                " à travers les semestres. Plus la couleur est foncée, plus le niveau de maîtrise attendu est élevé."
            ], style={'fontSize': '16px', 'marginBottom': '20px'}),
            dcc.Graph(figure=create_personal_roadmap(), config={'displayModeBar': True})
        ])
    
    elif tab == 'tab-workload':
        return html.Div([
            html.H2("Prédiction de Votre Charge de Travail", 
                    style={'color': COLORS['warning'], 'marginBottom': '20px'}),
            html.P([
                "Anticipez les semestres chargés ! Les barres montrent le ",
                html.Strong("nombre d'objectifs d'apprentissage par niveau"),
                ", et la ligne rouge indique le nombre de ",
                html.Strong("compétences uniques à maîtriser"),
                "."
            ], style={'fontSize': '16px', 'marginBottom': '20px'}),
            dcc.Graph(figure=create_workload_prediction(), config={'displayModeBar': True})
        ])
    
    elif tab == 'tab-radar':
        return html.Div([
            html.H2("Équilibre de Vos Compétences", 
                    style={'color': COLORS['info'], 'marginBottom': '20px'}),
            html.P([
                "Ce radar montre si votre formation est ",
                html.Strong("équilibrée entre les différents domaines"),
                " (TC-1 à TC-4 et IDU-1 à IDU-4). Un curriculum équilibré forme un cercle régulier."
            ], style={'fontSize': '16px', 'marginBottom': '20px'}),
            dcc.Graph(figure=create_competency_radar(), config={'displayModeBar': True})
        ])
    
    elif tab == 'tab-critical':
        return html.Div([
            html.H2("Les Compétences Critiques à Maîtriser", 
                    style={'color': COLORS['danger'], 'marginBottom': '20px'}),
            html.P([
                "Les compétences listées ici sont les ",
                html.Strong("plus fréquentes dans votre curriculum"),
                ". Plus elles apparaissent souvent, plus elles sont importantes pour votre réussite. La couleur indique le niveau de maîtrise attendu."
            ], style={'fontSize': '16px', 'marginBottom': '20px'}),
            dcc.Graph(figure=create_critical_path(), config={'displayModeBar': True})
        ])
    
    elif tab == 'tab-matrix':
        return html.Div([
            html.H2("Carte Module-Compétence", 
                    style={'color': COLORS['success'], 'marginBottom': '20px'}),
            html.P([
                "Cette matrice révèle ",
                html.Strong("quels modules développent quelles compétences"),
                ". Parfait pour comprendre où vous allez apprendre chaque compétence spécifique."
            ], style={'fontSize': '16px', 'marginBottom': '20px'}),
            dcc.Graph(figure=create_module_competency_matrix(), config={'displayModeBar': True})
        ])
    
    elif tab == 'tab-sunburst':
        return html.Div([
            html.H2("Vue Hiérarchique du Curriculum", 
                    style={'color': COLORS['secondary'], 'marginBottom': '20px'}),
            html.P([
                "Cette vue en éclat solaire montre ",
                html.Strong("l'organisation hiérarchique"),
                " de votre formation : Semestre → UE → Modules → Compétences. Cliquez pour explorer !"
            ], style={'fontSize': '16px', 'marginBottom': '20px'}),
            dcc.Graph(figure=create_semester_comparison_sunburst(), config={'displayModeBar': True})
        ])


# ========== LANCEMENT DE L'APPLICATION ==========

if __name__ == '__main__':
    # Mode debug désactivé pour compatibilité Python 3.14+
    app.run(debug=False, port=8050, host='127.0.0.1')