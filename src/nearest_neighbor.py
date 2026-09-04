import numpy as np

from environnement_vrp import EnvironnementVRP


def nearest_neighbor(env):
    """
    Construit une tournée avec la méthode du plus proche voisin
    en tenant compte des clients disponibles dynamiquement.
    """

    position = 0

    visites = np.zeros(env.n_clients, dtype=bool)
    visites[0] = True

    capacite_restante = env.capacite_vehicule
    temps_actuel = 0.0

    distance_totale = 0.0
    nombre_retours_depot = 0
    nombre_etapes = 0

    clients_disponibles = env.clients_disponibles.copy()

    while not visites.all() and nombre_etapes < 500:

        # Mise à jour des clients disponibles
        nouveaux_clients = (
            (~clients_disponibles)
            & (env.df["temps_apparition"].values <= temps_actuel)
        )

        clients_disponibles = clients_disponibles | nouveaux_clients

        # Clients non visités, disponibles et compatibles avec la capacité
        clients_non_visites = np.where(
            clients_disponibles & (~visites)
        )[0]

        # On enlève le dépôt
        clients_non_visites = clients_non_visites[
            clients_non_visites != 0
        ]

        clients_valides = [
            client
            for client in clients_non_visites
            if env.df.loc[client, "demande"] <= capacite_restante
        ]

        # Aucun client disponible pouvant être servi
        if len(clients_valides) == 0:

            # S'il reste des clients qui vont apparaître plus tard
            clients_futurs = env.df[
                (~clients_disponibles)
                & (~visites)
                & (env.df["temps_apparition"] > temps_actuel)
            ]

            if len(clients_futurs) > 0:

                # On attend l'arrivée du prochain client
                prochain_temps = clients_futurs[
                    "temps_apparition"
                ].min()

                temps_actuel = prochain_temps

                clients_disponibles = (
                    clients_disponibles
                    | (
                        env.df["temps_apparition"].values
                        <= temps_actuel
                    )
                )

                continue

            # Sinon, retour au dépôt
            if position != 0:

                distance = env.matrice_distances[position][0]

                distance_totale += distance
                temps_actuel += distance

                position = 0
                capacite_restante = env.capacite_vehicule

                nombre_retours_depot += 1
                nombre_etapes += 1

                continue

            else:
                break

        # Choisir le client le plus proche
        client_choisi = min(
            clients_valides,
            key=lambda client:
            env.matrice_distances[position][client]
        )

        # Distance vers le client
        distance = env.matrice_distances[
            position
        ][client_choisi]

        distance_totale += distance
        temps_actuel += distance

        # Mise à jour
        position = client_choisi

        capacite_restante -= env.df.loc[
            client_choisi, "demande"
        ]

        visites[client_choisi] = True

        nombre_etapes += 1

    # Retour final au dépôt
    if position != 0:

        distance = env.matrice_distances[position][0]

        distance_totale += distance
        temps_actuel += distance

        nombre_etapes += 1

    return {
        "clients_visites": int(visites.sum() - 1),
        "clients_non_visites": int((~visites).sum()),
        "retours_depot": nombre_retours_depot,
        "etapes": nombre_etapes,
        "distance_totale": distance_totale,
        "temps_total": temps_actuel
    }


if __name__ == "__main__":

    print("\n==========================================")
    print(" NEAREST NEIGHBOR")
    print("==========================================")

    instances_test = [
        "RC101",
        "RC102",
        "RC103"
    ]

    DOD = 0.3

    resultats_tous = {}

    for nom_instance in instances_test:

        print("\n==========================================")
        print(f" INSTANCE : {nom_instance}")
        print("==========================================")

        env = EnvironnementVRP(
            instances_disponibles=[nom_instance],
            dod=DOD
        )

        env.instance_actuelle = nom_instance

        env.reset()

        resultats = nearest_neighbor(env)

        resultats_tous[nom_instance] = resultats

        print(
            f"Clients visités : "
            f"{resultats['clients_visites']}"
        )

        print(
            f"Clients non visités : "
            f"{resultats['clients_non_visites']}"
        )

        print(
            f"Retours au dépôt : "
            f"{resultats['retours_depot']}"
        )

        print(
            f"Nombre d'étapes : "
            f"{resultats['etapes']}"
        )

        print(
            f"Distance totale : "
            f"{resultats['distance_totale']:.2f}"
        )

        print(
            f"Temps total : "
            f"{resultats['temps_total']:.2f}"
        )

    print("\n\n==========================================")
    print(" RÉSUMÉ NEAREST NEIGHBOR")
    print("==========================================")

    for nom_instance, resultats in resultats_tous.items():

        print(
            f"\n{nom_instance} : "
            f"distance = "
            f"{resultats['distance_totale']:.2f}, "
            f"temps = "
            f"{resultats['temps_total']:.2f}, "
            f"clients visités = "
            f"{resultats['clients_visites']}, "
            f"retours dépôt = "
            f"{resultats['retours_depot']}"
        )
