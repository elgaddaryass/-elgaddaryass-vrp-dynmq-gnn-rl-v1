"""
etape 1/2  Environnement DVRPTW : combine les vraies données Solomon avec un Degré de Dynamisme (DoD) / validation /matrice de distance /

"""

import vrplib
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist


#PARTIE1:Charger une vraie instance Solomon
def charger_instance(nom_instance):   #Charge un fichier Solomon déjà téléchargé dans ../data/
    
    chemin_fichier = f"../data/{nom_instance}.txt"
    data = vrplib.read_instance(chemin_fichier, instance_format="solomon")
    return data


def instance_vers_dataframe(data):  #Transforme les données Solomon en tableau, avec TOUTES les colonnes utiles.
    
    n = len(data["node_coord"])
    df = pd.DataFrame({
        "client_id": range(n),
        "x": data["node_coord"][:, 0],              #car même eandna x,y,ready_time... kola haja bo7dha mais vrplib fsh katsharjehom kaykono nodecord(x,y)  time_window.....
        "y": data["node_coord"][:, 1],
        "demande": data["demand"],
        "ready_time": data["time_window"][:, 0],    #l'heure la plus tôt à laquelle tu as le droit de livrer ce client
        "due_time": data["time_window"][:, 1],      #l'heure limite pour le livrer
        "service_time": data["service_time"],       #le temps que ça prend pour livrer une fois sur place
        "connu_des_le_depart": True,
        "temps_apparition": 0.0 ,    #0.0    car on a round au dessous dnc il faut un nombre reel (avec verguel)
    })
    return df



# PARTIE2:Appliquer un Degré de Dynamisme (DoD)
def appliquer_degre_dynamisme(df, dod=0.2, seed=42):
    """
    Rend une partie des clients "dynamiques" (apparaissant en cours de route),
    en choisissant PARMI LES VRAIS CLIENTS SOLOMON (pas des clients inventés).

    dod : proportion de clients dynamiques (0.0 = tout statique, 1.0 = tout dynamique)
    """
    np.random.seed(seed)
    df = df.copy()

    clients_ids = df[df["client_id"] != 0]["client_id"].values  # on exclut le dépôt (id=0)
    n_dynamique = int(len(clients_ids) * dod)    # dik int ghat7ydna lfassila 
    clients_dynamiques = np.random.choice(clients_ids, size=n_dynamique, replace=False)      #replacce=false  (bash mayhzsh shi whda meawda)

    for client_id in clients_dynamiques:
        idx = df[df["client_id"] == client_id].index[0]  # retrouver sa ligne exacte dans le tableau.  (index  :donne  numéro de position invisible hana 7it id bdina 0 1 2 ghaykon id o idx b7al b7al mais ila kan id mkhrbeq  idx ghykon b7al position de chque ligne dans excel))
        due_time = df.loc[idx, "due_time"]                   #loc(ligne.colonne)= kanakhdo dik lvaleur li fihom

        # le client apparaît à un instant aléatoire, mais AVANT la moitié de sa propre deadline
        temps_apparition = np.random.uniform(0, due_time * 0.5)

        df.loc[idx, "connu_des_le_depart"] = False    #acceder  ltbl l ligne idx o collone coon_des_le_depart o dir l valeur li ela lessr
        df.loc[idx, "temps_apparition"] = round(temps_apparition, 1)           #tmps_appr feh bzzf d lfasila ,dnc round(a,1) dit arrondie ce nombre en 1 seul nolbre apres la vérgule

    dod_reel = n_dynamique / len(clients_ids)         #dik n_dyn fash 7sbnaha DoD hzina bla fassilaa donc had DoD reel nissba kbira ykon mashi DoD L3adi mn gher ila 7sbna flwel o dik n_dynm khrjat bla fassila qbel mandiro int
    print(f"DoD demandé : {dod} | DoD réel : {dod_reel:.2f} | "
          f"{n_dynamique} clients dynamiques sur {len(clients_ids)}")

    return df



#PARTIE 3:Valider l'instance générée
def valider_instance(df, capacite_vehicule, n_vehicules):
    """Vérifie que l'instance générée est réaliste et faisable."""
    erreurs = []

    demande_totale = df[df["client_id"] != 0]["demande"].sum()
    capacite_totale_flotte = capacite_vehicule * n_vehicules
    if demande_totale > capacite_totale_flotte:
        erreurs.append(f"Demande totale ({demande_totale}) > capacité flotte ({capacite_totale_flotte})")

    fenetres_invalides = df[df["ready_time"] >= df["due_time"]]
    if len(fenetres_invalides) > 0:
        erreurs.append(f"{len(fenetres_invalides)} fenêtres de temps invalides (ready >= due)")

    clients_dynamiques = df[df["connu_des_le_depart"] == False]
    incoherents = clients_dynamiques[clients_dynamiques["temps_apparition"] >= clients_dynamiques["due_time"]]
    if len(incoherents) > 0:
        erreurs.append(f"{len(incoherents)} clients apparaissent après leur propre deadline")

    demande_trop_grande = df[df["demande"] > capacite_vehicule]
    if len(demande_trop_grande) > 0:
        erreurs.append(f"{len(demande_trop_grande)} clients ont une demande > capacité d'un véhicule")

    if df.isna().sum().sum() > 0:  #df.isna katredna tableau bmeme taille mais false true ila kant shi valeur manquante (video ou nan) katkon true o shi lakher false . sum() lwla katdir la somme dkola colone (true=1 false 0 ) , sum tanya kattdir somme ldak ligne li bqa ola lcln
       erreurs.append("Des valeurs manquantes ont été détectées dans le tableau")


    valide = len(erreurs) == 0
    print(f"\n=== Validation ===")
    print(f"Instance valide : {valide}")
    for e in erreurs:
        print(f"  ERREUR : {e}")

    return valide





def calculer_matrice_distances(df):
    """Calcule la distance euclidienne entre chaque paire de clients."""
    coords = df[["x", "y"]].values
    matrice_distances = cdist(coords, coords, metric="euclidean")
    print(f"Matrice de distances calculée : {matrice_distances.shape}")        #shape at3tena taille d matrice 
    return matrice_distances

# =========================================================
# EXÉCUTION
# =========================================================

if __name__ == "__main__":              #Lance ce qui suit seulement si ce fichier est exécuté directement.
   
    instances = ["C101", "C102", "C103", "C104", "R101", "R102", "R103", "R104",
                 "RC101", "RC102", "RC103", "RC104"]

    for nom_instance in instances:
        print(f"\n########## {nom_instance} ##########")

        #Charger une vraie instance Solomon
        data = charger_instance(nom_instance)
        df = instance_vers_dataframe(data)
        print(df.head())         #df.head   show the 5 first ligne in the table

        #Valider (capacité=200, 25 véhicules -> valeurs de l'en-tête du fichier de cette instace (exemple C101 au debut))
        for dod in [0.1, 0.3, 0.5]:
            print(f"\n=========== DoD = {dod} ===========")
            df_dynamique = appliquer_degre_dynamisme(df, dod=dod, seed=42)

            #Valider (capacité=200, 25 véhicules -> valeurs de l'en-tête du fichier de cette instace (exemple C101 au debut))
            valider_instance(df_dynamique, capacite_vehicule=data["capacity"], n_vehicules=25)
            matrice_distances = calculer_matrice_distances(df_dynamique)
            






















