
from dash import dcc, html, ctx
from dash.dependencies import Input, Output, State

from app_core import (
    app,
    df,
    competency_counts_spark,
    semesters,
    competency_counts_bar,
    df_pivot_global,
    df_pivot_semester,
    df_pivot_module,
    get_sorted_competencies,
    create_sparkline,
    create_heatmap_for_semester,
    create_heatmap_for_global,
    create_heatmap_for_module,
    create_vertical_stacked_bar_chart_for_competency,
    create_stacked_bar_chart_for_competency,
    update_network,
)

# 1) Sparklines sous le réseau
@app.callback(
    Output('network-sparklines-container', 'children'),
    [Input('competency-category', 'value')]
)
def update_sparklines(input_value):
    options = get_sorted_competencies(input_value)
    cards = []
    for comp in options:
        comp_name = comp['value']
        if comp_name in competency_counts_spark:
            spark = create_sparkline(competency_counts_spark[comp_name], semesters)
            cards.append(html.Div([
                html.Div(comp_name, style={'textAlign': 'center', 'fontWeight': 'bold', 'marginBottom': '5px', 'fontSize': '12px'}),
                html.Div(spark, style={'width': '120px', 'height': '30px', 'overflow': 'hidden', 'whiteSpace': 'normal'}),
            ], style={'display': 'inline-block', 'margin': '10px', 'textAlign': 'center'}))
        else:
            cards.append(html.Div())
    return cards

# 2) Heatmaps par semestre (liste)
@app.callback(
    Output('semester-heatmaps-container', 'children'),
    [Input('color-mode-toggle', 'value')]
)
def display_semesters(color_mode):
    out = []
    semesters_sorted = sorted(df['Semestre'].dropna().unique())
    for k in semesters_sorted:
        semester_graph = create_heatmap_for_semester(k, df_pivot_semester, ylegend=False)
        out.append(html.Div([
            html.Div(f"Semestre {k}", style={'textAlign': 'center', 'fontWeight': 'bold', 'marginBottom': '5px', 'fontSize': '12px'}),
            html.Div(dcc.Graph(figure=semester_graph), style={'width': '1000px', 'height': '500px', 'overflow': 'hidden', 'whiteSpace': 'normal'}),
        ], style={'display': 'inline-block', 'margin': '10px', 'textAlign': 'center'}))
    if not out:
        out.append(html.Div())
    return out

# 3) Toggle affichage légende
@app.callback(
    [Output("legend-content", "style"), Output("toggle-legend", "children")],
    [Input("toggle-legend", "n_clicks"), Input("legend-visible", "data")],
    State("legend-content", "style")
)
def toggle_legend_and_visibility(n_clicks, is_visible, current_style):
    if n_clicks % 2 == 1 or not is_visible:
        return {'display': 'none'}, "▶"
    return {'display': 'block'}, "▼"

# Dans app_callbacks.py

# 4) Options checklist avec mini-graphes
@app.callback(
    Output('competency-list', 'options'),
    # NOTE: J'ai enlevé 'color-mode-toggle' des Inputs pour le moment pour simplifier.
    # On pourra le remettre plus tard si nécessaire.
    [Input('competency-category', 'value')]
)
def update_checklist_with_sparklines(selected_category):
    competency_options = get_sorted_competencies(selected_category)
    checklist_items = []

    # Le mode couleur est maintenant géré directement ici. 'color' est plus visuel.
    color_mode = 'color' 

    for comp in competency_options:
        comp_name = comp['value']
        
        # Nous créons maintenant le graphique vertical pour chaque compétence
        if comp_name in competency_counts_bar:
            vertical_bar_chart = create_vertical_stacked_bar_chart_for_competency(
                competency_counts_bar, comp_name, color_mode
            )
        else:
            # S'il n'y a pas de données, on affiche un espace vide
            vertical_bar_chart = html.Div()

        # Ceci est la nouvelle structure de chaque "label" dans la checklist.
        # C'est un composant Dash complexe, pas juste du texte.
        checklist_items.append({
            'label': html.Div([
                # Le nom de la compétence
                html.Span(comp_name, style={'fontWeight': 'bold'}),
                # Le graphique, centré sous le nom
                html.Div(vertical_bar_chart, style={'width': '95%', 'margin': 'auto'})
            ], style={'padding': '5px', 'borderBottom': '1px solid #eee'}),
            'value': comp_name,
        })
        
    return checklist_items

# 5) Graph principal (réseau / heatmaps + navigation)
@app.callback(
    [Output('network-graph', 'figure'), Output('network-graph', 'config'), Output('heatmaps-data', 'data')],
    [Input('heatmap-or-forest', 'value'), Input('network-graph', 'clickData'), Input('go_back', 'n_clicks'), Input('heatmaps-data', 'data'), Input('competency-category', 'value'), Input('competency-list', 'value')]
)
def update_graph(heatmap_or_forest, clickData, go_back, data, selected_category, selected_competencies):
    stored_data = data
    if heatmap_or_forest == "Forest":
        graph = update_network(selected_category, selected_competencies)
        return graph, None, stored_data

    # Mode Heatmaps
    graph = None
    if clickData is not None and ctx.triggered_id != "go_back":
        stored_data['level'] = min(max(stored_data['level'] + 1, 0), 2)
        y_val = clickData["points"][0]["y"]
    if ctx.triggered_id == "go_back":
        stored_data['level'] = min(max(stored_data['level'] - 1, 0), 2)

    if stored_data['level'] == 0:
        stored_data['module'] = None
        stored_data['semester'] = None
        graph = create_heatmap_for_global(df_pivot_global)
    elif stored_data['level'] == 1:
        stored_data['module'] = None
        if ctx.triggered_id != "go_back" and 'y_val' in locals():
            stored_data['semester'] = int(y_val)
        graph = create_heatmap_for_semester(stored_data['semester'], df_pivot_semester)
    else:
        # niveau 2 : module
        if 'y_val' in locals():
            stored_data['module'] = y_val
        graph = create_heatmap_for_module(stored_data['module'], df_pivot_module)

    config = {"staticPlot": stored_data["level"] == 2}
    return graph, config, stored_data


if __name__ == '__main__':
    app.run_server(debug=True)


# Dans app_callbacks.py

# ==============================================================================
# ----- NOUVEAU CALLBACK POUR LE CLASSEMENT DYNAMIQUE (Question 6) -----
# ==============================================================================

from app_core import create_competency_leaderboard_graph # On importe la fonction de dessin

@app.callback(
    Output('competency-leaderboard', 'figure'),
    [Input('competency-category', 'value')]
)
def update_leaderboard(selected_category):
    """
    Ce callback met à jour le graphique du classement en fonction de la catégorie
    de compétences sélectionnée (TC, IDU, ou ALL).
    """
    
    # 1. Filtrer les données en fonction de la sélection
    if selected_category == 'ALL':
        # Pas de filtre, on prend toutes les compétences
        filtered_competencies = df.explode("All_Competencies")
    else:
        # On filtre les compétences qui commencent par la catégorie sélectionnée (ex: 'TC-')
        exploded_df = df.explode("All_Competencies")
        filtered_competencies = exploded_df[exploded_df['All_Competencies'].str.startswith(selected_category, na=False)]

    # Sécurité : si le filtre ne retourne aucune compétence, on affiche un graphique vide
    if filtered_competencies.empty:
        # Ici, vous pourriez retourner une figure avec un message "Pas de données"
        return go.Figure().update_layout(title_text=f"Aucune compétence trouvée pour la catégorie {selected_category}")

    # 2. Recalculer le classement sur les données filtrées
    competency_counts = filtered_competencies["All_Competencies"].value_counts().reset_index()
    competency_counts.columns = ['Competency', 'Count']
    top_15 = competency_counts.head(15).sort_values(by='Count', ascending=True)

    # 3. Redessiner et retourner le graphique avec les nouvelles données
    return create_competency_leaderboard_graph(top_15)