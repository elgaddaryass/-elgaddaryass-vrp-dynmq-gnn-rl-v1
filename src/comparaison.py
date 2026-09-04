import numpy as np
import torch

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecNormalize
)

from stable_baselines3.common.monitor import Monitor

from environnement_vrp import EnvironnementVRP
from nearest_neighbor import nearest_neighbor


# ==========================================
# PARAMÈTRES
# ==========================================

INSTANCES_TEST = [
    "RC101",
    "RC102",
    "RC103"
]

DOD = 0.3
N_SCENARIOS = 10

CHEMIN_MODELE = "agent_vrp_temps_reel"
CHEMIN_NORMALISATION = "vecnormalize_temps_reel.pkl"
CHEMIN_GNN = "gnn_temps_reel.pth"


# ==========================================
# MASQUE PPO
# ==========================================

def masque_valide(env):
    return env.unwrapped.action_masks()


# ==========================================
# CRÉER UN SCÉNARIO
# ==========================================

def creer_scenario(nom_instance, seed):

    env = EnvironnementVRP(
        instances_disponibles=[nom_instance],
        dod=DOD
    )

    env.instance_actuelle = nom_instance

    # reset avec seed
    observation, info = env.reset(
        seed=seed
    )

    scenario = {
        "df": env.df.copy(deep=True),

        "matrice_distances":
            env.matrice_distances.copy(),

        "graphe":
            env.graphe,

        "donnees_gnn":
            env.donnees_gnn,

    }

    return scenario


# ==========================================
# INSTALLER SCÉNARIO
# ==========================================

def installer_scenario(env, scenario):

    env.df = scenario["df"].copy(
        deep=True
    )

    env.matrice_distances = (
        scenario["matrice_distances"].copy()
    )

    env.graphe = scenario["graphe"]

    env.donnees_gnn = scenario["donnees_gnn"]

    env.embeddings = (
       env.gnn(
          env.donnees_gnn.x,
          env.donnees_gnn.edge_index
       )
       .detach()
       .numpy()
    )
    env.n_clients = len(env.df)

    env.position_actuelle = 0
    env.temps_actuel = 0.0

    env.capacite_restante = (
        env.capacite_vehicule
    )

    env.visites = np.zeros(
        env.n_clients,
        dtype=bool
    )

    env.visites[0] = True

    env.clients_disponibles = (
        env.df["connu_des_le_depart"]
        .values.copy()
    )

    env.clients_disponibles[0] = True




# ==========================================
# DIAGNOSTIC TRAJECTOIRE PPO
# ==========================================

def diagnostiquer_ppo(
    nom_instance,
    scenario
):

    print("\n==========================================")
    print(" DIAGNOSTIC PPO")
    print("==========================================")

    # ======================================
    # ENVIRONNEMENT
    # ======================================

    env_base = EnvironnementVRP(
        instances_disponibles=[nom_instance],
        dod=DOD
    )

    env_base.instance_actuelle = nom_instance

    # ======================================
    # CHARGER GNN
    # ======================================

    env_base.gnn.load_state_dict(
        torch.load(
            CHEMIN_GNN,
            map_location="cpu"
        )
    )

    env_base.gnn.eval()

    # ======================================
    # INSTALLER SCÉNARIO
    # ======================================

    installer_scenario(
        env_base,
        scenario
    )

    # ======================================
    # WRAPPERS
    # ======================================

    env = Monitor(env_base)

    env = ActionMasker(
        env,
        masque_valide
    )

    env = DummyVecEnv(
        [lambda: env]
    )

    # ======================================
    # NORMALISATION
    # ======================================

    env = VecNormalize.load(
        CHEMIN_NORMALISATION,
        env
    )

    env.training = False
    env.norm_reward = False

    # ======================================
    # CHARGER PPO
    # ======================================

    modele = MaskablePPO.load(
        CHEMIN_MODELE,
        env=env
    )

    # ======================================
    # RESET
    # ======================================

    env.reset()

    env_base = env.envs[0].unwrapped

    # ======================================
    # REMETTRE LE SCÉNARIO
    # ======================================

    installer_scenario(
        env_base,
        scenario
    )

    # ======================================
    # OBSERVATION INITIALE
    # ======================================

    observation_brute = (
        env_base._construire_observation()
    )

    observation = env.normalize_obs(
        observation_brute.reshape(1, -1)
    )

    # ======================================
    # BOUCLE
    # ======================================

    termine = False
    etape = 0

    while not termine and etape < 30:

        position_avant = (
            env_base.position_actuelle
        )

        temps_avant = (
            env_base.temps_actuel
        )

        capacite_avant = (
            env_base.capacite_restante
        )

        masque = env_base.action_masks()

        actions_valides = np.where(masque)[0]

        action, _ = modele.predict(
            observation,
            action_masks=masque,
            deterministic=True
        )

        action = int(action[0])

        print(
            f"\nÉtape {etape + 1}"
        )

        print(
            f"  Position actuelle : {position_avant}"
        )

        print(
            f"  Temps : {temps_avant:.2f}"
        )

        print(
            f"  Capacité : {capacite_avant:.2f}"
        )

        print(
            f"  Actions valides : "
            f"{actions_valides[:20]}"
        )

        print(
            f"  Action PPO : {action}"
        )

        if action == 0:
            print(
                "  >>> PPO choisit le DÉPÔT"
            )
        else:
            print(
                f"  >>> PPO choisit le client {action}"
            )

        # ==================================
        # STEP
        # ==================================

        observation, recompense, termine_array, infos = (
            env.step(
                np.array([action])
            )
        )

        termine = bool(
            termine_array[0]
        )

        observation = observation

        print(
            f"  Récompense : "
            f"{float(recompense[0]):.2f}"
        )

        print(
            f"  Nouvelle position : "
            f"{env_base.position_actuelle}"
        )

        print(
            f"  Nouveau temps : "
            f"{env_base.temps_actuel:.2f}"
        )

        print(
            f"  Nouvelle capacité : "
            f"{env_base.capacite_restante:.2f}"
        )

        etape += 1









# ==========================================
# ANALYSE DES VALEURS PPO
# ==========================================

def analyser_valeurs_ppo(
    nom_instance,
    scenario
):

    print("\n==========================================")
    print(" ANALYSE DES VALEURS PPO")
    print("==========================================")

    # ======================================
    # ENVIRONNEMENT
    # ======================================

    env_base = EnvironnementVRP(
        instances_disponibles=[nom_instance],
        dod=DOD
    )

    env_base.instance_actuelle = nom_instance

    # ======================================
    # CHARGER GNN
    # ======================================

    env_base.gnn.load_state_dict(
        torch.load(
            CHEMIN_GNN,
            map_location="cpu"
        )
    )

    env_base.gnn.eval()

    installer_scenario(
        env_base,
        scenario
    )

    # ======================================
    # WRAPPERS
    # ======================================

    env = Monitor(env_base)

    env = ActionMasker(
        env,
        masque_valide
    )

    env = DummyVecEnv(
        [lambda: env]
    )

    env = VecNormalize.load(
        CHEMIN_NORMALISATION,
        env
    )

    env.training = False
    env.norm_reward = False

    # ======================================
    # CHARGER PPO
    # ======================================

    modele = MaskablePPO.load(
        CHEMIN_MODELE,
        env=env
    )

    env.reset()

    env_base = env.envs[0].unwrapped

    installer_scenario(
        env_base,
        scenario
    )

    # ======================================
    # OBSERVATION INITIALE
    # ======================================

    observation_brute = (
        env_base._construire_observation()
    )

    observation = env.normalize_obs(
        observation_brute.reshape(1, -1)
    )

    # ======================================
    # ÉTAPE 1
    # ======================================

    masque = env_base.action_masks()

    action, _ = modele.predict(
        observation,
        action_masks=masque,
        deterministic=True
    )

    action = int(action[0])

    print("\n------------------------------------------")
    print(" ÉTAPE 1 : DEPUIS LE DÉPÔT")
    print("------------------------------------------")

    print(
        f"Action PPO : Client {action}"
    )

    # ======================================
    # EXÉCUTER PREMIÈRE ACTION
    # ======================================

    observation, recompense, termine, infos = (
        env.step(
            np.array([action])
        )
    )

    # ======================================
    # NOUVEL ÉTAT
    # ======================================

    position = env_base.position_actuelle

    print(
        f"Nouvelle position : Client {position}"
    )

    print(
        f"Temps : {env_base.temps_actuel:.2f}"
    )

    print(
        f"Capacité restante : "
        f"{env_base.capacite_restante:.2f}"
    )

    # ======================================
    # OBSERVATION APRÈS CLIENT
    # ======================================

    observation_brute = (
        env_base._construire_observation()
    )

    observation = env.normalize_obs(
        observation_brute.reshape(1, -1)
    )

    # ======================================
    # NOUVEAU MASQUE
    # ======================================

    masque = env_base.action_masks()

    actions_valides = np.where(masque)[0]

    # ======================================
    # DISTRIBUTION PPO
    # ======================================

    obs_tensor = torch.as_tensor(
        observation,
        dtype=torch.float32,
        device=modele.device
    )

    distribution = (
        modele.policy.get_distribution(
            obs_tensor
        )
    )

    probabilites = (
        distribution.distribution.probs
        .detach()
        .cpu()
        .numpy()[0]
    )

    # ======================================
    # ACTIONS VALIDES
    # ======================================

    valeurs = [
        (
            int(a),
            float(probabilites[a])
        )
        for a in actions_valides
    ]

    valeurs.sort(
        key=lambda x: x[1],
        reverse=True
    )

    # ======================================
    # AFFICHAGE
    # ======================================

    print("\n------------------------------------------")
    print(" ÉTAPE 2 : DEPUIS LE CLIENT")
    print("------------------------------------------")

    print(
        f"Position actuelle : Client {position}"
    )

    print(
        f"Nombre d'actions valides : "
        f"{len(actions_valides)}"
    )

    print("\nTop 15 actions selon PPO :")

    for rang, (a, p) in enumerate(
        valeurs[:15],
        start=1
    ):

        if a == 0:
            nom = "DÉPÔT"
        else:
            nom = f"Client {a}"

        print(
            f"{rang:2d}. "
            f"{nom:<12} "
            f"probabilité = {p:.6f}"
        )

    # ======================================
    # PROBABILITÉ DU DÉPÔT
    # ======================================

    if masque[0]:

        print(
            f"\n>>> Probabilité du DÉPÔT : "
            f"{probabilites[0]:.6f}"
        )

    else:

        print(
            "\n>>> Le dépôt n'est pas valide."
        )

    # ======================================
    # ACTION DÉTERMINISTE
    # ======================================

    action_suivante, _ = modele.predict(
        observation,
        action_masks=masque,
        deterministic=False
    )

    action_suivante = int(
        action_suivante[0]
    )

    print(
        f"\n>>> Action choisie par PPO : "
        f"{action_suivante}"
    )

    if action_suivante == 0:

        print(
            ">>> PPO retourne effectivement "
            "au DÉPÔT."
        )

    else:

        print(
            f">>> PPO continue vers "
            f"le client {action_suivante}."
        )















# ==========================================
# ÉVALUATION PPO
# ==========================================

def evaluer_ppo(
    nom_instance,
    scenario
):

    # ======================================
    # ENVIRONNEMENT DE BASE
    # ======================================

    env_base = EnvironnementVRP(
        instances_disponibles=[nom_instance],
        dod=DOD
    )

    env_base.instance_actuelle = (
        nom_instance
    )

    # ======================================
    # CHARGER LE GNN DE L'ENTRAÎNEMENT
    # ======================================

    env_base.gnn.load_state_dict(
        torch.load(
            CHEMIN_GNN,
            map_location="cpu"
        )
    )

    env_base.gnn.eval()

    # ======================================
    # INSTALLER LE SCÉNARIO
    # ======================================

    installer_scenario(
        env_base,
        scenario
    )

    # ======================================
    # WRAPPERS
    # ======================================

    env = Monitor(env_base)

    env = ActionMasker(
        env,
        masque_valide
    )

    env = DummyVecEnv(
        [lambda: env]
    )

    # ======================================
    # NORMALISATION
    # ======================================

    env = VecNormalize.load(
        CHEMIN_NORMALISATION,
        env
    )

    env.training = False
    env.norm_reward = False

    # ======================================
    # MODÈLE
    # ======================================

    modele = MaskablePPO.load(
        CHEMIN_MODELE,
        env=env
    )

    # ======================================
    # RESET AUTOMATIQUE DU WRAPPER
    # ======================================

    env.reset()

    env_base = (
        env.envs[0].unwrapped
    )

    # ======================================
    # REMETTRE LE SCÉNARIO EXACT
    # ======================================

    installer_scenario(
        env_base,
        scenario
    )

    # ======================================
    # OBSERVATION INITIALE
    # ======================================

    observation_brute = (
        env_base._construire_observation()
    )

    observation_brute = (
        observation_brute.reshape(1, -1)
    )

    observation = env.normalize_obs(
        observation_brute
    )

    # ======================================
    # VARIABLES
    # ======================================

    termine = False
    tronque = False

    distance_totale = 0.0
    attente_totale = 0.0

    recompense_totale = 0.0

    nombre_retours_depot = 0
    nombre_etapes = 0

    clients_visites = set()

    # ======================================
    # BOUCLE PPO
    # ======================================

    while (
        not termine
        and not tronque
        and nombre_etapes < 500
    ):

        position_avant = (
            env_base.position_actuelle
        )

        # ==================================
        # MASQUE
        # ==================================

        action_mask = (
            env_base.action_masks()
        )

        # ==================================
        # ACTION PPO
        # ==================================

        action, _ = modele.predict(
            observation,
            action_masks=action_mask,
            deterministic=True
        )

        action = int(action[0])

        # ==================================
        # DISTANCE
        # ==================================

        if action != position_avant:

            distance = (
                env_base.matrice_distances[
                    position_avant
                ][action]
            )

            distance_totale += distance

        # ==================================
        # CLIENT VISITÉ
        # ==================================

        if action != 0:

            clients_visites.add(
                action
            )

        # ==================================
        # RETOUR AU DÉPÔT
        # ==================================

        if (
            action == 0
            and position_avant != 0
        ):

            nombre_retours_depot += 1

        # ==================================
        # ACTION DANS ENVIRONNEMENT
        # ==================================

        observation, recompense, termine_array, infos = (
            env.step(
                np.array([action])
            )
        )

        termine = bool(
            termine_array[0]
        )

        recompense_totale += float(
            recompense[0]
        )

        # ==================================
        # ATTENTE DYNAMIQUE
        # ==================================

        info = infos[0]

        if "attente" in info:

            attente_totale += float(
                info["attente"]
            )

        nombre_etapes += 1

    # ======================================
    # RÉSULTATS
    # ======================================

    nombre_clients_total = (
        env_base.n_clients - 1
    )

    nombre_clients_visites = (
        len(clients_visites)
    )

    clients_non_visites = (
        nombre_clients_total
        - nombre_clients_visites
    )

    # Le temps = déplacement + attente
    temps_total = (
        distance_totale
        + attente_totale
    )

    return {

        "clients_visites":
            nombre_clients_visites,

        "clients_non_visites":
            clients_non_visites,

        "retours_depot":
            nombre_retours_depot,

        "etapes":
            nombre_etapes,

        "distance_totale":
            distance_totale,

        "attente_totale":
            attente_totale,

        "temps_total":
            temps_total,

        "recompense":
            recompense_totale
    }


# ==========================================
# ÉVALUATION NEAREST NEIGHBOR
# ==========================================

def evaluer_nearest_neighbor(
    nom_instance,
    scenario
):

    env = EnvironnementVRP(
        instances_disponibles=[nom_instance],
        dod=DOD
    )

    env.instance_actuelle = (
        nom_instance
    )

    installer_scenario(
        env,
        scenario
    )

    return nearest_neighbor(env)


# ==========================================
# PROGRAMME PRINCIPAL
# ==========================================

if __name__ == "__main__":

    print("\n==========================================")
    print(" COMPARAISON PPO VS NEAREST NEIGHBOR")
    print("==========================================")

    print(
        f"\nDoD utilisé : {DOD}"
    )

    print(
        f"Nombre de scénarios : "
        f"{N_SCENARIOS}"
    )

    resultats_globaux = {}

    # ======================================
    # INSTANCES
    # ======================================

    # ======================================
    # DIAGNOSTIC D'UN SEUL SCÉNARIO
    # ======================================

    scenario = creer_scenario(
        "RC101",
        1000
    )

    analyser_valeurs_ppo(
        "RC101",
        scenario
    )

    exit()
    
    
    for nom_instance in INSTANCES_TEST:

        print("\n==========================================")
        print(
            f" INSTANCE : {nom_instance}"
        )
        print("==========================================")

        resultats_ppo = []
        resultats_nn = []

        # ==================================
        # SCÉNARIOS
        # ==================================

        for numero in range(
            N_SCENARIOS
        ):

            seed = 1000 + numero

            print(
                f"\n--- Scénario "
                f"{numero + 1}/{N_SCENARIOS} "
                f"(seed={seed}) ---"
            )

            # ==================================
            # CRÉER UN SEUL SCÉNARIO
            # ==================================

            scenario = creer_scenario(
                nom_instance,
                seed
            )

            # ==================================
            # PPO
            # ==================================

            print(
                "Évaluation PPO..."
            )

            resultat_ppo = evaluer_ppo(
                nom_instance,
                scenario
            )

            resultats_ppo.append(
                resultat_ppo
            )

            print(
                f"PPO : distance = "
                f"{resultat_ppo['distance_totale']:.2f}, "
                f"temps = "
                f"{resultat_ppo['temps_total']:.2f}, "
                f"clients = "
                f"{resultat_ppo['clients_visites']}, "
                f"retours = "
                f"{resultat_ppo['retours_depot']}"
            )

            # ==================================
            # NEAREST NEIGHBOR
            # ==================================

            print(
                "Évaluation Nearest Neighbor..."
            )

            resultat_nn = (
                evaluer_nearest_neighbor(
                    nom_instance,
                    scenario
                )
            )

            resultats_nn.append(
                resultat_nn
            )

            print(
                f"NN  : distance = "
                f"{resultat_nn['distance_totale']:.2f}, "
                f"temps = "
                f"{resultat_nn['temps_total']:.2f}, "
                f"clients = "
                f"{resultat_nn['clients_visites']}, "
                f"retours = "
                f"{resultat_nn['retours_depot']}"
            )

        # ======================================
        # MOYENNES
        # ======================================

        moyenne_ppo_distance = np.mean([
            r["distance_totale"]
            for r in resultats_ppo
        ])

        moyenne_nn_distance = np.mean([
            r["distance_totale"]
            for r in resultats_nn
        ])

        moyenne_ppo_temps = np.mean([
            r["temps_total"]
            for r in resultats_ppo
        ])

        moyenne_nn_temps = np.mean([
            r["temps_total"]
            for r in resultats_nn
        ])

        moyenne_ppo_retours = np.mean([
            r["retours_depot"]
            for r in resultats_ppo
        ])

        moyenne_nn_retours = np.mean([
            r["retours_depot"]
            for r in resultats_nn
        ])

        moyenne_ppo_clients = np.mean([
            r["clients_visites"]
            for r in resultats_ppo
        ])

        moyenne_nn_clients = np.mean([
            r["clients_visites"]
            for r in resultats_nn
        ])

        # ======================================
        # SAUVEGARDE
        # ======================================

        resultats_globaux[nom_instance] = {

            "ppo_distance":
                moyenne_ppo_distance,

            "nn_distance":
                moyenne_nn_distance,

            "ppo_temps":
                moyenne_ppo_temps,

            "nn_temps":
                moyenne_nn_temps,

            "ppo_retours":
                moyenne_ppo_retours,

            "nn_retours":
                moyenne_nn_retours,

            "ppo_clients":
                moyenne_ppo_clients,

            "nn_clients":
                moyenne_nn_clients
        }

        # ======================================
        # AFFICHAGE
        # ======================================

        print(
            "\n------------------------------------------"
        )

        print(
            f"MOYENNES - {nom_instance}"
        )

        print(
            "------------------------------------------"
        )

        print(
            f"PPO - Distance moyenne : "
            f"{moyenne_ppo_distance:.2f}"
        )

        print(
            f"NN  - Distance moyenne : "
            f"{moyenne_nn_distance:.2f}"
        )

        print(
            f"PPO - Temps moyen : "
            f"{moyenne_ppo_temps:.2f}"
        )

        print(
            f"NN  - Temps moyen : "
            f"{moyenne_nn_temps:.2f}"
        )

        print(
            f"PPO - Retours moyens : "
            f"{moyenne_ppo_retours:.2f}"
        )

        print(
            f"NN  - Retours moyens : "
            f"{moyenne_nn_retours:.2f}"
        )

    # ==========================================
    # RÉSUMÉ GLOBAL
    # ==========================================

    print(
        "\n\n=========================================="
    )

    print(
        " RÉSUMÉ GLOBAL"
    )

    print(
        "=========================================="
    )

    print(
        "\nInstance | PPO distance | NN distance | "
        "PPO temps | NN temps"
    )

    print(
        "------------------------------------------"
    )

    for nom_instance, r in (
        resultats_globaux.items()
    ):

        print(
            f"{nom_instance:<8} | "
            f"{r['ppo_distance']:>12.2f} | "
            f"{r['nn_distance']:>11.2f} | "
            f"{r['ppo_temps']:>9.2f} | "
            f"{r['nn_temps']:>8.2f}"
        )

    # ==========================================
    # MOYENNES GLOBALES
    # ==========================================

    moyenne_ppo_distance = np.mean([
        r["ppo_distance"]
        for r in resultats_globaux.values()
    ])

    moyenne_nn_distance = np.mean([
        r["nn_distance"]
        for r in resultats_globaux.values()
    ])

    moyenne_ppo_temps = np.mean([
        r["ppo_temps"]
        for r in resultats_globaux.values()
    ])

    moyenne_nn_temps = np.mean([
        r["nn_temps"]
        for r in resultats_globaux.values()
    ])

    moyenne_ppo_retours = np.mean([
        r["ppo_retours"]
        for r in resultats_globaux.values()
    ])

    moyenne_nn_retours = np.mean([
        r["nn_retours"]
        for r in resultats_globaux.values()
    ])

    # ==========================================
    # COMPARAISON
    # ==========================================

    difference = (
        moyenne_ppo_distance
        - moyenne_nn_distance
    )

    pourcentage = (
        difference
        / moyenne_nn_distance
    ) * 100

    print(
        "\n=========================================="
    )

    print(
        " MOYENNES GLOBALES"
    )

    print(
        "=========================================="
    )

    print("\nPPO :")

    print(
        f"Distance moyenne : "
        f"{moyenne_ppo_distance:.2f}"
    )

    print(
        f"Temps moyen : "
        f"{moyenne_ppo_temps:.2f}"
    )

    print(
        f"Retours dépôt moyens : "
        f"{moyenne_ppo_retours:.2f}"
    )

    print("\nNearest Neighbor :")

    print(
        f"Distance moyenne : "
        f"{moyenne_nn_distance:.2f}"
    )

    print(
        f"Temps moyen : "
        f"{moyenne_nn_temps:.2f}"
    )

    print(
        f"Retours dépôt moyens : "
        f"{moyenne_nn_retours:.2f}"
    )

    print(
        "\n=========================================="
    )

    print(
        " ANALYSE"
    )

    print(
        "=========================================="
    )

    print(
        f"\nDifférence moyenne de distance : "
        f"{difference:.2f}"
    )

    print(
        f"Différence en pourcentage : "
        f"{pourcentage:.2f}%"
    )

    if moyenne_ppo_distance < moyenne_nn_distance:

        print(
            "\nPPO obtient une distance moyenne "
            "plus faible que Nearest Neighbor."
        )

    else:

        print(
            "\nNearest Neighbor obtient une distance "
            "moyenne plus faible que PPO."
        )