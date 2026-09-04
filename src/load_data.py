import vrplib





instances = ["C101", "C102", "C103","C104", "R101", "R102", "R103","R104", "RC101", "RC102", "RC103","RC104"]

toutes_les_donnees = {}
for nom in instances:
  
    chemin_fichier = f"../data/{nom}.txt"
    toutes_les_donnees[nom] = vrplib.read_instance(chemin_fichier, instance_format="solomon")

print("Instances téléchargées :", list(toutes_les_donnees.keys()))
