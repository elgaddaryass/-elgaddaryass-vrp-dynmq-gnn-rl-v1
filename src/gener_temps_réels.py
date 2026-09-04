
import numpy as np
import pandas as pd
print("Le script a démarré !")

#1er recette
def generer_donnees_initiales(n_clients=20, taille_zone=100, capacite=100,
                                demande_min=1, demande_max=20, seed=42):
    """
    Génère la carte de clients au départ (avant que la tournée ne commence).
    """
    np.random.seed(seed)

    depot = {
        "client_id": 0,
        "x": taille_zone / 2,
        "y": taille_zone / 2,
        "demande": 0,
        "temps_apparition": 0,   
    }

    clients = {
        "client_id": range(1, n_clients + 1),
        "x": np.random.uniform(0, taille_zone, n_clients),
        "y": np.random.uniform(0, taille_zone, n_clients),
        "demande": np.random.randint(demande_min, demande_max, n_clients),
        "temps_apparition": [0] * n_clients,   #tous connus dès le départ
    }
    df_clients = pd.DataFrame(clients)
    df = pd.concat([pd.DataFrame([depot]), df_clients], ignore_index=True)

    print(f"Carte initiale générée : {n_clients} clients connus dès le départ (temps=0)")
    return df      # gnr la carte de départ



#2eme recette
def generer_nouveau_client(prochain_id, taille_zone, temps_actuel,
                             demande_min=1, demande_max=20):
    """
    Crée UN nouveau client qui "apparaît" à un instant donné pendant la simulation.
    """
    return {
        "client_id": prochain_id,
        "x": np.random.uniform(0, taille_zone),
        "y": np.random.uniform(0, taille_zone),
        "demande": np.random.randint(demande_min, demande_max),
        "temps_apparition": temps_actuel,
    }

#3eme recette
def simuler_evenements(df_initial, n_pas_de_temps=50, probabilite_evenement=0.1,
                         taille_zone=100, seed=123):
    """
    Simule le déroulement du temps : à chaque pas de temps, un nouveau client
    peut apparaître selon une probabilité donnée.

    n_pas_de_temps : nombre d'unités de temps simulées (ex: 50 "minutes" de tournée)
    probabilite_evenement : chance qu'un nouveau client apparaisse à CHAQUE pas de temps
    """
    np.random.seed(seed)

    prochain_id = df_initial["client_id"].max() + 1
    nouveaux_clients = []

    for temps_actuel in range(1, n_pas_de_temps + 1):
        if np.random.random() < probabilite_evenement:
            nouveau_client = generer_nouveau_client(prochain_id, taille_zone, temps_actuel)
            nouveaux_clients.append(nouveau_client)
            print(f"[Temps {temps_actuel}] Nouveau client apparu : id={prochain_id}, "
                  f"position=({nouveau_client['x']:.1f}, {nouveau_client['y']:.1f}), "
                  f"demande={nouveau_client['demande']}")
            prochain_id += 1

    if nouveaux_clients:
        df_nouveaux = pd.DataFrame(nouveaux_clients)
        df_complet = pd.concat([df_initial, df_nouveaux], ignore_index=True)
    else:
        df_complet = df_initial.copy()
        print("Aucun événement dynamique ne s'est produit pendant cette simulation.")

    print(f"\nTotal : {len(df_initial)} clients initiaux + {len(nouveaux_clients)} apparus en cours de route "
          f"= {len(df_complet)} clients au total")

    return df_complet     # gnr d'événements dynamiques 




if __name__ == "__main__":
    
    df_initial = generer_donnees_initiales(n_clients=20, seed=42)

    print("\n--- Aperçu de la carte initiale ---")
    print(df_initial.head())

    print("\n--- Simulation du temps réel (50 pas de temps) ---")
    df_final = simuler_evenements(
        df_initial,
        n_pas_de_temps=50,
        probabilite_evenement=0.1,   #10%  chance à chaque pas de temps
        seed=123
    )

    print("\n--- Aperçu final (avec les nouveaux clients ajoutés) ---")
    print(df_final.tail())    # execution