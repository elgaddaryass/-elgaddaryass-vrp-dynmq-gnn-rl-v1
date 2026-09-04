import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from environnement_vrp import EnvironnementVRP
from nearest_neighbor import nearest_neighbor


def masque_valide(env):
    return env.unwrapped.action_masks()


def charger_environnement():
    """
    Charge l'environnement avec exactement les paramètres
    utilisés pour l'évaluation du PPO.
    """

    env = EnvironnementVRP(
        ["RC101"],
        dod=0.3
    )

    env = Monitor(env)
    env = ActionMasker(env, masque_valide)
    env = DummyVecEnv([lambda: env])

    env = VecNormalize.load(
        "vecnormalize_temps_reel.pkl",
        env
    )

    env.training = False
    env.norm_reward = False

    return env


def executer_ppo(env, seed=1000):
    """
    Exécute une tournée avec le modèle PPO
    et enregistre les clients visités.
    """

    modele = MaskablePPO.load(
        "agent_vrp_temps_reel",
        env=env
    )

    environnement = env.venv.envs[0].unwrapped

    # Initialisation déterministe du scénario
    np.random.seed(seed)
    observation, info = environnement.reset()

    route = [0]
    recompense_totale = 0.0
    nombre_etapes = 0

    for _ in range(500):

        masque = environnement.action_masks()

        # Vérification utile pour détecter un problème de masque
        if not masque.any():
            print("⚠️ Aucun client valide dans le masque.")
            break

        observation_normalisee = env.normalize_obs(
            observation.reshape(1, -1)
        )

        action, _ = modele.predict(
            observation_normalisee,
            action_masks=masque,
            deterministic=True
        )

        action = int(action.item())

        ancienne_position = environnement.position_actuelle
        ancien_temps = environnement.temps_actuel

        observation, recompense, termine, tronque, info = (
            environnement.step(action)
        )

        recompense_totale += recompense
        nombre_etapes += 1

        # Enregistrer uniquement les changements de position
        if action != ancienne_position:

            if action == 0:
                route.append(0)

            elif action != 0:
                route.append(action)

        if termine or tronque:
            break

    return {
        "route": route,
        "distance": calculer_distance(environnement, route),
        "temps": environnement.temps_actuel,
        "recompense": recompense_totale,
        "etapes": nombre_etapes,
        "clients_visites": environnement.visites.sum() - 1,
        "clients_non_visites": (
            len(environnement.visites)
            - environnement.visites.sum()
        ),
        "environnement": environnement
    }


def calculer_distance(env, route):
    """
    Calcule la distance totale à partir de la route.
    """

    distance_totale = 0.0

    for i in range(len(route) - 1):

        depart = route[i]
        arrivee = route[i + 1]

        distance_totale += env.matrice_distances[
            depart
        ][arrivee]

    return distance_totale


def executer_nearest_neighbor(env):
    """
    Exécute Nearest Neighbor sur le même scénario.
    """

    resultat = nearest_neighbor(env)

    return resultat


def afficher_route(nom, route):
    print("\n" + "=" * 60)
    print(nom)
    print("=" * 60)

    print("Nombre de positions :", len(route))
    print("Nombre de retours dépôt :", route.count(0) - 1)

    print("\nRoute :")

    # Afficher par morceaux pour éviter une ligne énorme
    taille = 20

    for i in range(0, len(route), taille):
        morceau = route[i:i + taille]
        print(" → ".join(map(str, morceau)))


def visualiser_route(env, route_ppo, route_nn):
    """
    Affiche graphiquement les deux tournées.
    """

    df = env.df

    plt.figure(figsize=(12, 8))

    # Tous les clients
    plt.scatter(
        df["x"],
        df["y"],
        s=25,
        label="Clients"
    )

    # Dépôt
    plt.scatter(
        df.loc[0, "x"],
        df.loc[0, "y"],
        s=150,
        marker="s",
        label="Dépôt"
    )

    # Route PPO
    for i in range(len(route_ppo) - 1):

        a = route_ppo[i]
        b = route_ppo[i + 1]

        plt.plot(
            [df.loc[a, "x"], df.loc[b, "x"]],
            [df.loc[a, "y"], df.loc[b, "y"]],
            linewidth=1.2,
            alpha=0.7
        )

    plt.title("Tournée PPO - RC101 - DoD 0.3")
    plt.xlabel("Coordonnée X")
    plt.ylabel("Coordonnée Y")
    plt.legend()
    plt.grid(True)

    plt.show()


if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("DIAGNOSTIC DE LA TOURNÉE PPO")
    print("=" * 60)

    # -------------------------------------------------
    # 1. Charger environnement + modèle PPO
    # -------------------------------------------------

    env = charger_environnement()

    # -------------------------------------------------
    # 2. Exécuter PPO
    # -------------------------------------------------

    print("\nÉvaluation PPO...")

    resultat_ppo = executer_ppo(
        env,
        seed=1000
    )

    environnement = resultat_ppo["environnement"]

    # -------------------------------------------------
    # 3. Afficher résultat PPO
    # -------------------------------------------------

    afficher_route(
        "TOURNÉE PPO",
        resultat_ppo["route"]
    )

    print("\nDistance PPO :",
          round(resultat_ppo["distance"], 2))

    print("Temps PPO :",
          round(resultat_ppo["temps"], 2))

    print("Récompense PPO :",
          round(resultat_ppo["recompense"], 2))

    print("Étapes PPO :",
          resultat_ppo["etapes"])

    print("Clients visités :",
          resultat_ppo["clients_visites"])

    print("Clients non visités :",
          resultat_ppo["clients_non_visites"])

    # -------------------------------------------------
    # 4. Exécuter Nearest Neighbor
    # -------------------------------------------------

    print("\nÉvaluation Nearest Neighbor...")

    resultat_nn = executer_nearest_neighbor(
        environnement
    )

    print("\nDistance NN :",
          round(resultat_nn["distance_totale"], 2))

    print("Temps NN :",
          round(resultat_nn["temps_total"], 2))

    print("Étapes NN :",
          resultat_nn["etapes"])

    print("Clients visités NN :",
          resultat_nn["clients_visites"])

    # -------------------------------------------------
    # 5. Comparaison
    # -------------------------------------------------

    print("\n" + "=" * 60)
    print("COMPARAISON")
    print("=" * 60)

    print(
        "PPO :",
        round(resultat_ppo["distance"], 2)
    )

    print(
        "NN  :",
        round(resultat_nn["distance_totale"], 2)
    )

    difference = (
        resultat_ppo["distance"]
        - resultat_nn["distance_totale"]
    )

    print(
        "Différence :",
        round(difference, 2)
    )

    if resultat_nn["distance_totale"] > 0:

        pourcentage = (
            difference
            / resultat_nn["distance_totale"]
        ) * 100

        print(
            "Écart relatif :",
            round(pourcentage, 2),
            "%"
        )

    # -------------------------------------------------
    # 6. Visualisation
    # -------------------------------------------------

    visualiser_route(
        environnement,
        resultat_ppo["route"],
        None
    )