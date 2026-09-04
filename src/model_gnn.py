"""
Étape 4 : GNN — extraction de caractéristiques avec GraphSAGE
Choix justifié : GraphSAGE gère mieux les graphes qui changent de taille,
ce qui correspond à l'aspect dynamique du projet (clients ajoutés en cours de route).
"""
 
import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv
 
from dvrptw_dynamique import (
    charger_instance,
    instance_vers_dataframe,
    appliquer_degre_dynamisme,
    valider_instance,
    calculer_matrice_distances,
)
from construction_graphe import construire_graphe
from conversion_gnn import preparer_donnees_gnn
 
 

# Le modèle GNN : 2 couches GraphSAGE
 
class GNNSimple(nn.Module):
    def __init__(self, nb_caracteristiques_entree, taille_cachee, taille_embedding):
        super().__init__()
        self.couche1 = SAGEConv(nb_caracteristiques_entree, taille_cachee)
        self.couche2 = SAGEConv(taille_cachee, taille_embedding)
        self.activation = nn.ReLU()
 
    def forward(self, x, edge_index):
        x = self.couche1(x, edge_index)
        x = self.activation(x)
        x = self.couche2(x, edge_index)
        return x
 
 

# EXÉCUTION
if __name__ == "__main__":
    nom_instance = "C101"
    data_solomon = charger_instance(nom_instance)
    df = instance_vers_dataframe(data_solomon)
 
    df_dynamique = appliquer_degre_dynamisme(df, dod=0.3, seed=42)
    valider_instance(df_dynamique, capacite_vehicule=data_solomon["capacity"], n_vehicules=25)
    matrice_distances = calculer_matrice_distances(df_dynamique)
 
    graphe = construire_graphe(df_dynamique, matrice_distances, k=8)
    donnees_gnn = preparer_donnees_gnn(graphe)
 
    modele = GNNSimple(
        nb_caracteristiques_entree=donnees_gnn.num_node_features,
        taille_cachee=16,
        taille_embedding=8,
    )
 
    embeddings = modele(donnees_gnn.x, donnees_gnn.edge_index)
 
    print(f"\nForme des données AVANT le GNN : {donnees_gnn.x.shape}")
    print(f"Forme des embeddings APRÈS le GNN : {embeddings.shape}")
 
    print("\n--- Embedding du dépôt (nœud 0), AVANT ---")
    print(donnees_gnn.x[0])
    print("--- Embedding du dépôt (nœud 0), APRÈS le GNN (GraphSAGE) ---")
    print(embeddings[0])
 