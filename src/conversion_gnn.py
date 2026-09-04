"""
&tape 4:Conversion du graphe NetworkX vers PyTorch Geometric ( préparation)
"""

import torch
import numpy as np
from torch_geometric.utils import from_networkx


from dvrptw_dynamique import (
    charger_instance,
    instance_vers_dataframe,
    appliquer_degre_dynamisme,
    valider_instance,
    calculer_matrice_distances,)

from construction_graphe import (
    construire_graphe,
)



# Conversion du graphe vers le format PyTorch Geometric
def preparer_donnees_gnn(graphe):
    """
    Convertit un graphe NetworkX en objet PyTorch Geometric,
    en choisissant quelles infos du client deviennent des "features" numériques.
    """
    # PyTorch Geometric a besoin de savoir quels attributs des nœuds utiliser comme features
    data = from_networkx(
        graphe,
        group_node_attrs=["x", "y", "demande", "ready_time", "due_time", "service_time"]
    )
    data.x = data.x.float()   #7it ymken ykono b vergul o maytenregistrawssh entier o homa reel
    print(f"Données converties : {data.num_nodes} nœuds, {data.num_edges} arêtes")
    print(f"Chaque nœud a {data.num_node_features} caractéristiques (features)")
    return data





# EXÉCUTION
if __name__ == "__main__":
  
    nom_instance = "C101"
    data_solomon = charger_instance(nom_instance)
    df = instance_vers_dataframe(data_solomon)

    df_dynamique = appliquer_degre_dynamisme(df, dod=0.3, seed=42)
    valider_instance(df_dynamique, capacite_vehicule=data_solomon["capacity"], n_vehicules=25)
    matrice_distances = calculer_matrice_distances(df_dynamique)

    graphe = construire_graphe(df_dynamique, matrice_distances, k=8)


    donnees_gnn = preparer_donnees_gnn(graphe)   # conversion pour PyTorch Geometric

    print("\n--- Aperçu des features du premier nœud ---")
    print(donnees_gnn.x[0])   # les 6 valeurs (x, y, demande, ready_time, due_time, service_time)