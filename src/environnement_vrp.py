"""
Étape 6:Environnement de simulation (Gymnasium) /l'agent RL évoluera dans cet environnement à l'étape suivante.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from dvrptw_dynamique import (
    charger_instance,
    instance_vers_dataframe,
    appliquer_degre_dynamisme,
    calculer_matrice_distances,
)
from construction_graphe import construire_graphe
from conversion_gnn import preparer_donnees_gnn
from model_gnn import GNNSimple


class EnvironnementVRP(gym.Env):
    """
    Environnement DVRPTW : un véhicule doit visiter des clients,
    en respectant la capacité et les fenêtres de temps.
    """

    def __init__(self,instances_disponibles,dod=0.3,k_voisins=8,capacite_vehicule=200,n_clients_max=101):
        super().__init__()
        self.instances_disponibles = instances_disponibles
        self.dod = dod
        self.k_voisins = k_voisins
        self.capacite_vehicule = capacite_vehicule
        self.instance_actuelle = None


        # Le GNN utilisé pour transformer le graphe en embeddings (pas encore entraîné)
        self.gnn = GNNSimple(nb_caracteristiques_entree=6, taille_cachee=16, taille_embedding=8)

        # Réglages Gymnasium : à définir précisément une fois une instance chargée
        self.action_space = spaces.Discrete(n_clients_max)
        self.observation_space = spaces.Box(
           low=-np.inf,
           high=np.inf,
           shape=(8 * n_clients_max + 2 * n_clients_max + 3,),
           dtype=np.float32
        )
    
    # reset() : démarrer une nouvelle tournée      #Comme cliquer sur "Nouvelle partie" dans un jeu vidéo :
                                                        # On choisit une carte au hasard (une instance Solomon)
                                                        #Le véhicule est placé au dépôt, avec sa capacité pleine
                                                        #Personne n'a encore été visité
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        #1;Choisir une instance au hasard parmi celles disponibles (entraînement)
        if self.instance_actuelle is None:
            nom_instance = np.random.choice(self.instances_disponibles)
        else:
            nom_instance = self.instance_actuelle



        data_solomon = charger_instance(nom_instance)
        self.df = instance_vers_dataframe(data_solomon)
        self.df = appliquer_degre_dynamisme(self.df, dod=self.dod, seed=np.random.randint(0, 10000))

        #2;Construire le graphe et les embeddings
        self.matrice_distances = calculer_matrice_distances(self.df)
        self.graphe = construire_graphe(self.df, self.matrice_distances, k=self.k_voisins)
        self.donnees_gnn = preparer_donnees_gnn(self.graphe)
        self.embeddings = self.gnn(self.donnees_gnn.x, self.donnees_gnn.edge_index).detach().numpy()

        #3;Initialiser l'état de la tournée
        self.n_clients = len(self.df)
        self.position_actuelle = 0                 #on démarre au dépôt (id=0)
        self.temps_actuel = 0.0
        self.capacite_restante = self.capacite_vehicule
        self.visites = np.zeros(self.n_clients, dtype=bool)   #np.zeros donne une liste de 0 de taille selon le nombre donnéé / dtype:bool liste de true et false et 0 dnc false 
        self.visites[0] = True   #le dépôt est "déjà visité" (point de départ)  

        # Clients disponibles au début de la simulation
        self.clients_disponibles = (
        self.df["connu_des_le_depart"].values.copy()
        )

        #Le dépôt est toujours disponible
        self.clients_disponibles[0] = True
          
        

        observation = self._construire_observation()
        info = {}
        return observation, info          #Tu construis l'observation de départ (fonction qu'on regarde plus bas) et tu la renvoies — c'est ce que Gymnasium exige toujours en sortie de reset().

 
    # step(action) : le véhicule visite un nouveau client        Comme déplacer ton personnage vers une case du plateau :

                                                                          #  Tu dis "je vais chez le client numéro 12"
                                                                          # Le jeu vérifie : est-ce que c'est possible ? (le client n'est pas déjà visité, il reste assez de place dans le camion)
                                                                          # Si oui : le véhicule s'y déplace, le jeu calcule des points (bons ou mauvais)
                                                                          # Si non : grosse pénalité de points, comme une erreur


     
   
    def step(self, action):
        client_choisi = action
        recompense = 0.0
        termine = False
        info = {}   #reçois l'action choisie par l'agent

        # Vérification de faisabilité (l'agent a choisi un client invalide)
        
        if client_choisi == 0:

           # Si le véhicule est déjà au dépôt,
           # on attend l'arrivée du prochain client.
           if self.position_actuelle == 0:

              clients_futurs = self.df[
                 (~self.clients_disponibles)
                & (
                    self.df["temps_apparition"]
                   > self.temps_actuel
                )
              ]

              if len(clients_futurs) > 0:

                prochain_temps = clients_futurs[
                   "temps_apparition"
                ].min()

                self.temps_actuel = prochain_temps

                nouveaux_clients = (
                   self._mettre_a_jour_clients_disponibles()
                )

                info["attente"] = float(prochain_temps)

                info["nouveaux_clients"] = (
                   nouveaux_clients.tolist()
                )

                recompense = -1.0

              else:
                 recompense = 0.0

           else:

               distance = self.matrice_distances[
                 self.position_actuelle
               ][0]

               self.temps_actuel += distance

               self.position_actuelle = 0

               self.capacite_restante = (
                  self.capacite_vehicule
               )

               nouveaux_clients = (
                  self._mettre_a_jour_clients_disponibles()
               )

               info["nouveaux_clients"] = (
                   nouveaux_clients.tolist()
               )

               recompense = -distance
        elif self.visites[client_choisi]:
            recompense = -50.0   #forte pénalité:client déjà visité
        elif self.df.loc[client_choisi, "demande"] > self.capacite_restante: 
            recompense = -50.0   #forte pénalité:capacité insuffisante
        else:
            # Déplacement valide
            distance = self.matrice_distances[self.position_actuelle][client_choisi]
            self.temps_actuel += distance
            self.capacite_restante -= self.df.loc[client_choisi, "demande"]
            self.position_actuelle = client_choisi
            self.visites[client_choisi] = True

            # De nouveaux clients peuvent apparaître après le déplacement
            nouveaux_clients = self._mettre_a_jour_clients_disponibles()

            if len(nouveaux_clients) > 0:
             info["nouveaux_clients"] = nouveaux_clients.tolist()
            else:
             info["nouveaux_clients"] = []




            recompense = -distance   #on pénalise la distance parcourue

            # Bonus/malus liés à la fenêtre de temps
            due_time = self.df.loc[client_choisi, "due_time"]
            if self.temps_actuel <= due_time:
                recompense += 5.0    #bonus:livré dans les temps
            else:
                recompense -= 20.0   #malus:livré en retard

        #Fin de l'épisode si tous les clients ont été visités
        if self.visites.all():
            termine = True

        observation = self._construire_observation()
        return observation, recompense, termine, False, info    #renvoi le nouveau etat .......

    


    
    
    
    
    
    
    
    

    
    
    
    
    
    
    
    def _mettre_a_jour_clients_disponibles(self):
       """
       Rend disponibles les clients dont le temps d'apparition
       est atteint.
       """

       temps_apparition = self.df["temps_apparition"].values

       nouveaux_clients = (
          (~self.clients_disponibles)
           & (temps_apparition <= self.temps_actuel)
       )

       self.clients_disponibles = (
         self.clients_disponibles | nouveaux_clients
       )

       return np.where(nouveaux_clients)[0]
    
    
    
    
    
    
    
    
    
    
    

    # Fonctions utilitaires
    
    def _construire_observation(self):
       """
       Construit l'observation complète pour l'agent PPO.

       L'agent reçoit :
       - les embeddings GNN de tous les clients
       - les clients actuellement disponibles
       - les clients déjà visités
       - l'état du véhicule
       """

       # Embeddings GNN de tous les clients
       embeddings_tous_clients = self.embeddings.flatten()

       # Disponibilité des clients
       disponibilite = self.clients_disponibles.astype(np.float32)

       # Clients déjà visités
       visites = self.visites.astype(np.float32)

       # État du véhicule
       etat_vehicule = np.array([
           self.capacite_restante / self.capacite_vehicule,
           self.temps_actuel / 1000.0,
           self.visites.sum() / max(1, self.n_clients - 1),
       ], dtype=np.float32)

       observation = np.concatenate([
           embeddings_tous_clients,
           disponibilite,
           visites,
           etat_vehicule
       ])

       return observation.astype(np.float32)

   

    

    def action_masks(self):
      """
      Masque les actions impossibles :
      - client déjà visité
      - client pas encore apparu
      - demande supérieure à la capacité restante
      """

      masque = (
        self.clients_disponibles
        & (~self.visites)
      )

      demandes = self.df["demande"].values

      masque = masque & (
        demandes <= self.capacite_restante
      )

      # Le dépôt est disponible uniquement si le véhicule
      # est actuellement chez un client.
      if self.position_actuelle != 0:
        masque[0] = True
      else:
        masque[0] = False

      return masque

# EXÉCUTION : test rapide de l'environnement
if __name__ == "__main__":
    instances_entrainement = ["C101", "C102", "C103", "R101", "R102", "R103"]

    env = EnvironnementVRP(instances_entrainement, dod=0.3)

    observation, info = env.reset()

    print("\n--- TEST TEMPS RÉEL ---")

    print(
      "Clients disponibles au départ :",
      int(env.clients_disponibles.sum())
    )

    print(
      "Clients cachés au départ :",
      int((~env.clients_disponibles).sum())
    )

    clients_dynamiques = env.df[
       env.df["connu_des_le_depart"] == False
    ] 

    print("\nQuelques clients dynamiques :")
    print(
      clients_dynamiques[
         ["client_id", "temps_apparition", "connu_des_le_depart"]
      ].head(10)
    )


    print("\n--- TEST TEMPS RÉEL ---")

    # ... ton bloc actuel ...


    # ==================================================
    # TEST APPARITION DES CLIENTS
    # ==================================================

    print("\n--- TEST APPARITION DES CLIENTS ---")

    print("Temps actuel :", env.temps_actuel)

    clients_dynamiques = env.df[
      env.df["connu_des_le_depart"] == False
    ]

    premier_client = clients_dynamiques.iloc[0]

    print(
      f"Client choisi pour le test : "
      f"{int(premier_client['client_id'])}"
    )

    print(
      f"Temps d'apparition : "
      f"{premier_client['temps_apparition']}"
    )
 
    print(
      "Disponible avant :",
      env.clients_disponibles[
          int(premier_client["client_id"])
      ]
    )

    # On avance artificiellement le temps
    env.temps_actuel = premier_client["temps_apparition"]

    # Mise à jour des clients disponibles
    nouveaux_clients = env._mettre_a_jour_clients_disponibles()
  
    print("Temps actuel après attente :", env.temps_actuel)

    print(
       "Nouveaux clients apparus :",
       nouveaux_clients.tolist()
    )

    print(
       "Disponible après :",
          env.clients_disponibles[
          int(premier_client["client_id"])
        ]
    )



    print(f"\nObservation initiale, forme : {observation.shape}")

    
    for i in range(5):
        action = env.action_space.sample()
        observation, recompense, termine, tronque, info = env.step(action)
        print(f"Étape {i+1} : action={action}, récompense={recompense:.2f}, terminé={termine}")


    print("\n--- SIMULATION TEMPS RÉEL ---")

    env.reset()

    for i in range(20):
       masque = env.action_masks()
       clients_valides = np.where(masque)[0]

       print(
           f"Étape {i+1} | "
           f"Temps = {env.temps_actuel:.1f} | "
           f"Clients disponibles = {int(env.clients_disponibles.sum())} | "
           f"Clients visités = {int(env.visites.sum()-1)}"
       )

       if len(clients_valides) == 0:
           print("Aucune action valide.")
           break

       action = clients_valides[0]

       observation, recompense, termine, tronque, info = env.step(action)

       if info.get("nouveaux_clients"):
           print("  → Nouveaux clients :", info["nouveaux_clients"])

       if termine or tronque:
           print("Simulation terminée.")
           break     