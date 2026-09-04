"""
étape 3 de l'architecture  ( crée le graphe logistique )
"""

import networkx as nx
import numpy as np

from dvrptw_dynamique import (
    charger_instance,
    instance_vers_dataframe,
    appliquer_degre_dynamisme,
    valider_instance,
    calculer_matrice_distances,
)



# PARTIE 3 : Construction du graphe (k plus proches voisins)              (ce graphe est symétrique)
def construire_graphe(df, matrice_distances, k=8):
   
    G = nx.Graph()       #crée un grph vide

    #Étape 1:ajouter chaque client comme un nœud,avec toutes ses infos
    for _, ligne in df.iterrows():    # df.iterrows    mthd dans pandas kat3tina ligne par ligne l'index dyal dak ligne  + contenue  dyalo , dnc lboucle need 2 variables (index de ligne , son contenue , in first varib we don't push anything(_) because we don't need it)
        G.add_node(
            int(ligne["client_id"]),
            x=ligne["x"],
            y=ligne["y"],
            demande=ligne["demande"],
            ready_time=ligne["ready_time"],
            due_time=ligne["due_time"],
            service_time=ligne["service_time"],
            connu_des_le_depart=ligne["connu_des_le_depart"],
            temps_apparition=ligne["temps_apparition"],
        )


    n = len(df)
    #Étape 2:pour chaque client,trouver ses k plus proches voisins et créer un arêtt
    for i in range(n):
        distances_depuis_i = matrice_distances[i]     # matrice_distance[][]  donc    [i] nous donne une liste  de i de distances depuis tout les neoud vers i
        voisins_proches = np.argsort(distances_depuis_i)[1:k + 1]    #argost à un pricipe de savoire trier liste de plus petit vers le plus grand  , mais il donne une liste des indices  de la liste qu'on veux trier pour  que les valeur de la liste est trier  
                                                                     # ....[1:k+1]  Fash ghat3tina liste des indices o li ghykono les valeur dylha trié ghykon awl indice howa dyal lvaleur 0 nit distance dyal dk i moqaranatan brasso , dnc manhzosh 0 ta l k+1    ldk knn li bghena ,sh7al men voisin bghena 
        for j in voisins_proches:
            client_id_i = df.iloc[i]["client_id"]
            client_id_j = df.iloc[j]["client_id"]
            G.add_edge(int(client_id_i), int(client_id_j), weight=matrice_distances[i][j]) #ajouter arret entre ces deux client et le poid c'est la distance entre eux      (weight)

    print(f"Graphe construit : {G.number_of_nodes()} nœuds, {G.number_of_edges()} arêtes (k={k})")         #G.number_of_nodes(): nombre de noeuds
    return G


# =========================================================
# EXÉCUTION
# =========================================================

if __name__ == "__main__":          #Sans cette ligne : emprunter une fonction (fash nkono f shi ficjier khor o ndiro from hconstruction_graphe import shi fonction ) déclencherait TOUT le fichier, même les parties que tu ne voulais pas  /Avec cette ligne : emprunter une fonction ne déclenche QUE cette fonction, rien d'autre
     # On reprend exactement le même déroulement que dans dvrptw_dynamique.py,
    # mais cette fois on ajoute la construction du graphe à la fin

    nom_instance = "C101"
    print(f"\n########## {nom_instance} ##########")

    data = charger_instance(nom_instance)
    df = instance_vers_dataframe(data)

    dod = 0.3
    df_dynamique = appliquer_degre_dynamisme(df, dod=dod, seed=42)
    valider_instance(df_dynamique, capacite_vehicule=data["capacity"], n_vehicules=25)
    matrice_distances = calculer_matrice_distances(df_dynamique)

    # NOUVEAU : construction du graphe
    graphe = construire_graphe(df_dynamique, matrice_distances, k=8)

    # Petit aperçu pour vérifier
    print("\n--- Aperçu du nœud dépôt (id=0) ---")
    print(graphe.nodes[0])

    print("\n--- Voisins du dépôt dans le graphe ---")
    print(list(graphe.neighbors(0)))