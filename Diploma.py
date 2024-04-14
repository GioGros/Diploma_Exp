import re
from github import Github
import networkx as nx
import dgl
import torch
from pycparser import c_parser

def preprocess_code(code):
    code = code.replace('\r', '')
    code = re.sub(r'#.*\n', '\n', code)
    return code

# Остальной код без изменений

# Загрузка исходного кода с GitHub
repo_url = "https://github.com/GioGros/Diploma_Exp"
c_code = download_code_from_github('https://github.com/GioGros/Diploma_Exp')

# Предварительная обработка кода на C
c_code_processed = preprocess_code(c_code)

# Построение AST из обработанного кода на C
ast_tree = build_ast(c_code_processed)

# Построение графа потоков управления из AST
cfg_graph = build_graph_from_ast(ast_tree)

# Преобразование графа в DGL Graph для использования с GNN
dgl_graph = dgl.from_networkx(cfg_graph)

# Создание признаков узлов графа (замените это на реальные признаки)
features = torch.randn(dgl_graph.number_of_nodes(), 10)

# Создание модели GNN
class GNNModel(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super(GNNModel, self).__init__()
        self.conv1 = nn.GCNConv(in_feats, hidden_size)
        self.conv2 = nn.GCNConv(hidden_size, num_classes)

    def forward(self, graph, features):
        h = self.conv1(graph, features)
        h = torch.relu(h)
        h = self.conv2(graph, h)
        return h

# Создание модели GNN
model = GNNModel(in_feats=features.shape[1], hidden_size=64, num_classes=2)

# Передача графа и признаков через модель GNN
output = model(dgl_graph, features)

def detect_vulnerabilities(graph):
    vulnerabilities = []

    for node in graph.nodes():
        if is_vulnerable(node):
            vulnerability_info = {
                'description': 'Описание уязвимости',
                'type': 'Тип уязвимости',
                'detection_methods': 'Методы обнаружения',
                'remediation_recommendations': 'Рекомендации по устранению'
            }
            vulnerabilities.append(vulnerability_info)

    # Формирование отчета с документацией к уязвимостям
    generate_vulnerabilities_report(vulnerabilities)

    return vulnerabilities