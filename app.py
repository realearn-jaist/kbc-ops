import dash
from dash import dcc, html, Input, Output
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import base64
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from owlready2 import *
import gc

"""
Function for finding relations within the ontology show path from the input class to owl.Thing.
Function loop recursively until owl.Thing is found. The relations found are append to realtion_list.
This function is used with FoodOn ontology.

Input:  cls (owlready2 object) --> Class used as head in relation
        realtion_list (list) --> List of relation in the format of <h ,r, t>

Output: Flase
"""
def find_parents_with_relations(cls, relation_list):
    try:
        parents = cls.is_a
        for parent in parents:
            if parent != owl.Thing:
                relation_list.append([str(cls).split('obo.')[-1].split(')')[0], 'subclassOf', str(parent).split('obo.')[-1].split(')')[0]])
                find_parents_with_relations(parent, relation_list)
            else:
                relation_list.append([str(cls).split('obo.')[-1].split(')')[0], 'subclassOf', str(parent).split('obo.')[-1].split(')')[0]])
    except Exception as e:
       pass


ONTO_foodon = get_ontology(os.path.abspath('ontology/foodon-merged.owl')).load()
ONTO_helis = get_ontology(os.path.abspath('ontology/helis_v1.00.origin.owl')).load()


entity_prefix = {'helis': "http://www.fbk.eu/ontologies/virtualcoach#",
                        'foodon': 'http://purl.obolibrary.org/obo/'}
delimiter = {"helis":"#", "foodon": "/"}

"""
Function for garbage count bar graph plotting. 
The bar graph will show the difference between the test set size (blue) compare to count of garbages (orange).

Input:  onto (string) --> Name of ontology contain in the csv
        embed (sttring) --> Name of embedding contain in csv

Output: img_html1 (html.Img) --> Image of garbage count vs all test count
"""
def plot_garbage_prediction(onto, embed):
    csv_file = 'data/onto_embed.csv'
    df = pd.read_csv(csv_file)
    filtered_df = df[(df['onto'] == onto) & (df['embed'] == embed)]
    all_data = filtered_df['all'].values[0]
    infer_data = filtered_df['infer'].values[0]
    plt.figure(figsize=(8, 6))
    plt.bar(['Test set', 'Garbage'], [all_data, infer_data], color=['#9DF1F0', 'orange'], label='Garbage')
    plt.text(0, all_data, str(all_data), ha='center', va='bottom')
    plt.text(1, infer_data, str(infer_data), ha='center', va='bottom')
    plt.xlabel('Data Type')
    plt.ylabel('Count')
    plt.title(f'Garbages in prediction from {embed} - {onto}')
    buffer1 = BytesIO()
    plt.savefig(buffer1, format='png')
    buffer1.seek(0)
    plot_data1 = base64.b64encode(buffer1.getvalue()).decode()
    img_html1 = html.Img(src='data:image/png;base64,{}'.format(plot_data1))
    plt.clf()
    plt.close()
    return img_html1


"""
Function for plotting bar graph showing difference in rank and score of 2 classes, true class(green) and inferable/garbage class(orange).

Input:  names_numbers(list) --> List of tuple (class name, rank)
        numbers(list) --> List of score from prediction
        subset_names(list) --> List of garbage classes
        individual(string) --> Name of class/individual used for prediction
        onto(string) --> Name of ontology which used for determining relation type

Output: img_html1 (html.Img) --> Image of bar graph showing rank and score of true class and garbage.
"""
def plot_bar_graph(names_numbers, numbers, subset_names, individual, onto):
    names = [pair[0] for pair in names_numbers]
    values = [pair[1] for pair in names_numbers]
    colors = []
    labels = []
    bar_values = []
    for name, value, rank in zip(names, numbers, values):
        if name in subset_names:
            if name == subset_names[-1]:
                colors.append('#94F19C')
            else:
                colors.append('#FC865A')
            labels.append(f"Rank: {rank}")
            bar_values.append(value)
        else:
            colors.append('gray')
            labels.append('')
            bar_values.append(None)
    plt.figure(figsize=(8, 6))
    bars = plt.bar(range(len(names)), numbers, color=colors)
    for name, bar, value in zip(names, bars, bar_values):
        if value is not None:
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(f"{name}:{value}"),
                     ha='center', va='bottom')
    plt.xticks(range(len(names)), labels)
    relation_type = 'isA' if onto == 'helis' else 'subclassOf'
    plt.title(f"'{relation_type}' relations predictions of '{individual}'")
    plt.xlabel('Predicted classes')
    plt.ylabel('Prediction scores')
    # plt.tight_layout()
    buffer1 = BytesIO()
    plt.savefig(buffer1, format='png')
    buffer1.seek(0)
    plot_data1 = base64.b64encode(buffer1.getvalue()).decode()
    img_html1 = html.Img(src='data:image/png;base64,{}'.format(plot_data1))
    plt.clf()
    plt.close()
    return img_html1


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css'] #Custom stlyesheets for dash apps
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server  


dropdown_options_helis = [{'label': str(i), 'value': i} for i in range(1, 10)]
dropdown_options_foodon = [{'label': str(i), 'value': i} for i in range(1, 6)]
app.layout = html.Div([
    html.H1("KBCOps - Knowledge Base Completion Operations"),
    html.P(["This demo is a simple proof of concept according to the paper 'Are Embeddings All We Need for Knowledge Base Completion? Insights from Description Logicians' built to show that when doing prediction, there might be a chance to predict to implicit knowledge already contained in the ontology which is called 'Garbage' knowledge in our definition.\n"]),
    html.Br(),
    html.P(["The ",
            html.Span("blue", style={'color': '#9DF1F0'}),
            " node represents test individual/class for the prediction.\n"], style={'float': 'left'}),
    html.P(["The ",
            html.Span("green", style={'color': '#94F19C'}),
            " node represents true class of the test individual/class.\n"]),
    html.Br(),
    html.P(["The ",
            html.Span("orange", style={'color': '#FC865A'}),
            " node(s) represent inferable node that were predicted with higher score that the true class.\n"]),
    html.Br(),
    html.H3("Ontology"),
    html.Div([
        dcc.RadioItems(
            id='radio-buttons-1',
            options=[
            {'label': 'Helis', 'value': 'helis'},
            {'label': 'FoodOn', 'value': 'foodon'},
            ],
            value='helis',
            labelStyle={'display': 'inline-block', 'margin-right': '10px'}
        )
    ]),
    html.H3("Type of embedding"),
    html.Div([
        dcc.RadioItems(
            id='radio-buttons-2',
            options=[
            {'label': 'Onto2Vec', 'value': 'onto2vec'},
            {'label': 'OPA2Vec', 'value': 'opa2vec'},
            {'label': 'RDF2Vec', 'value': 'rdf2vec'},
            {'label': 'OWL2Vec*', 'value': 'owl2vecstar'}
        ],
            value='onto2vec',
            labelStyle={'display': 'inline-block', 'margin-right': '10px'}
        )
    ]),
    html.Div([
        html.H3(id='header'),
        dcc.RadioItems(
            id='radio-buttons-3',
            options=dropdown_options_helis,
            value=1,
            labelStyle={'display': 'inline-block', 'margin-right': '10px'}
        )
    ]),
    html.Div(id='image-display', style={'float': 'left'}),
    html.Div(id='output-graph1', style={'float': 'right'}),
    html.Br(),
    html.Div(id='output-graph2')
])
# ----------


"""
Function for updating radio button when ontology type is changed.
"""
@app.callback(
    [Output('header', 'children'),
     Output('radio-buttons-3', 'options'),
     Output('radio-buttons-3', 'value')],
    [Input('radio-buttons-1', 'value'),
     Input('radio-buttons-2', 'value')]
)
def update_options(radio1, _):
    if radio1 == 'helis':
      header_text = "Garbage individuals (ABox Completion) with most difference in rank from true class"
    else:
      header_text = "Garbage classes (TBox Completion) with most difference in rank from true class"
    options = dropdown_options_helis if radio1 == 'helis' else dropdown_options_foodon
    return header_text, options, 1



"""
Main function that will use the input to calculate and draw output graphs.

Input:  onto(string) --> Name of ontology
        embedding(string) --> Name of embedding
        individual_num(int) --> The number of instance/class the will be shown

Output  img_html1(html.Img) --> Image from plot_bar_graph
        img_html2(html.Img) --> Image of relations plotted using networkx with relation data
        img3(html.Img) --> Image from plot_garbage_prediction
"""
@app.callback(
    [Output(component_id='output-graph1', component_property='children'),
     Output(component_id='output-graph2', component_property='children'),
     Output(component_id='image-display', component_property='children')],
    [Input(component_id='radio-buttons-1', component_property='value'),
     Input(component_id='radio-buttons-2', component_property='value'),
     Input(component_id='radio-buttons-3', component_property='value'),]
)
def plot_inferable(onto, embedding, individual_num):
    gc.collect()
    onto_data = pd.read_csv(os.path.abspath(f"data/{onto}_{embedding}.csv"))
    img3 = plot_garbage_prediction(onto, embedding)
    ind_num = individual_num if onto == 'helis' else individual_num-1
    individual = [x for x in list(onto_data['Individual'].tolist())]
    predict_lst = [x for x in list(onto_data['Predicted'].tolist())]
    truth = [x for x in list(onto_data['True'].tolist())]
    predict_rank = [x for x in list(onto_data['Predicted_rank'].tolist())]
    true_rank = [x for x in list(onto_data['True_rank'].tolist())]
    predict_score =  [x for x in list(onto_data['Score_predict'].tolist())]
    true_score =  [x for x in list(onto_data['Score_true'].tolist())]
    onto_name = onto
    relation_type = 'isA' if onto == 'helis' else 'subclassOf'
    entity_uri = entity_prefix[onto]+individual[ind_num]
    onto = ONTO_helis if onto == 'helis' else ONTO_foodon
    # ----------

    """Find relations and put in relations list"""
    entity = onto.search(iri = entity_uri)[0]
    subs = entity.INDIRECT_is_a
    if onto_name == 'foodon':
        relations = []
        find_parents_with_relations(entity, relations)
    else:
        subs = sorted(list(subs), key=lambda sub: len(list(sub.INDIRECT_is_a)))
        subs = [str(sub).split('.')[-1] if str(sub) != 'owl.Thing' else str(sub) for sub in subs ]
        relations = list()
        for i in range(len(subs)-1):
            relations.append([subs[i+1], 'subclassOf', subs[i]])
        relations.append([str(entity).split('.')[-1], relation_type, subs[-1]])
    relations = [relation for relation in relations if relation[0] != relation[2]]
    sorted_data = [(predict_lst[ind_num], predict_rank[ind_num]), (truth[ind_num], true_rank[ind_num])]
    score = [predict_score[ind_num], true_score[ind_num]]
    show =  [predict_lst[ind_num],  truth[ind_num]]
    img_html1 = plot_bar_graph(sorted_data, score, show, individual[ind_num], onto_name)
    # ----------


    """Code chunk for plotting relations using networkx"""
    if onto_name == 'helis':
        plt.figure(figsize=(20, len(relations)*2))
    else:
        plt.figure(figsize=(15, len(relations)*1))
    plt.title(f"Relation from owl.Thing to '{str(entity).split('.')[-1]}'")
    G = nx.DiGraph()
    truth = [truth[ind_num]]
    individual = [individual[ind_num]]
    infer = [predict_lst[ind_num]]
    for rel in relations:
        source, relation, target = rel
        G.add_edge(source, target, label=relation)
        G.add_nodes_from([source, target])
    node_colors = ['gray' if node not in truth and node not in individual and node not in infer else '#94F19C' if node in truth else '#FC865A' if node in infer else '#9DF1F0' for node in G.nodes()]
    pos = nx.nx_pydot.graphviz_layout(G, prog='dot')
    nx.draw(G, pos, with_labels=True, node_size=1500, node_color=node_colors, font_size=12, font_weight="bold")
    for edge, label in nx.get_edge_attributes(G, 'label').items():
        x = (pos[edge[0]][0] + pos[edge[1]][0]) / 2
        y = (pos[edge[0]][1] + pos[edge[1]][1]) / 2
        plt.text(x, y, label, horizontalalignment='center', verticalalignment='center')
    plt.tight_layout()
    buffer2 = BytesIO()
    plt.savefig(buffer2, format='png')
    buffer2.seek(0)
    plot_data2 = base64.b64encode(buffer2.getvalue()).decode()
    img_html2 = html.Img(src='data:image/png;base64,{}'.format(plot_data2))
    plt.clf()
    plt.close()
    # ----------

    return img_html1, img_html2, img3



if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8050))
    app.run_server(debug=False, host='0.0.0.0', port=port)