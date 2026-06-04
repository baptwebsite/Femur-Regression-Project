import os
import json
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split

def main():
    # MODIFIE LE CHEMIN ICI SI NÉCESSAIRE
    root_dataset = "data/" 
    json_path = os.path.join(root_dataset, 'dataset_PatientSize.json')
    
    if not os.path.exists(json_path):
        print(f"Erreur : Le fichier {json_path} est introuvable.")
        return

    # 1. Charger le JSON d'indexation
    with open(json_path, 'r') as f:
        all_data = json.load(f)

    # 2. Recréer exactement le split par Patient (Identique au DataLoader)
    groups = defaultdict(list)
    for item in all_data:
        filename = os.path.basename(item['obj_path'])
        root_id = filename.split('_aug')[0].replace('.obj', '')
        groups[root_id].append(item)
    
    unique_root_ids = sorted(list(groups.keys()))
    
    # Séparation 80% Train/Val et 10% Test (Strictement identique)
    _, test_ids = train_test_split(
        unique_root_ids, 
        test_size=0.10, 
        random_state=42
    )

    # Filtrer pour ne garder que les vrais fémurs du Test Set (sans augmentation)
    test_meshes = [
        entry for rid in test_ids 
        for entry in groups[rid] 
        if '_aug' not in entry['obj_path']
    ]

    print(f"=== ANALYSE DU TEST SET ===")
    print(f"Nombre de patients uniques en Test : {len(test_ids)}")
    print(f"Nombre de maillages réels en Test   : {len(test_meshes)}\n")

    # 3. Extraction des tailles réelles (PatientSize)
    true_sizes = [item['PatientSize'] for item in test_meshes]

    # 4. Définition des tranches de taille de 5cm en 5cm (en mètres)
    bins = np.arange(1.45, 2.05, 0.05)
    counts = defaultdict(int)
    
    for size in true_sizes:
        found = False
        for i in range(len(bins) - 1):
            low = bins[i]
            high = bins[i+1]
            if low <= size < high:
                counts[f"{low:.2f}m - {high:.2f}m"] += 1
                found = True
                break
        if not found:
            if size < bins[0]:
                counts[f"< {bins[0]:.2f}m"] += 1
            elif size >= bins[-1]:
                counts[f">= {bins[-1]:.2f}m"] += 1

    # 5. Affichage propre de la distribution
    print("Distribution des tailles dans le jeu de test :")
    print("-" * 40)
    
    # Affichage si valeurs en dessous de 1m45
    if f"< {bins[0]:.2f}m" in counts:
        print(f"          < {bins[0]:.2f}m : {counts[f'< {bins[0]:.2f}m']} fémur(s)")
        
    # Affichage des tranches standards
    for i in range(len(bins) - 1):
        interval = f"{bins[i]:.2f}m - {bins[i+1]:.2f}m"
        print(f"  Entre {interval} : {counts[interval]} fémur(s)")
        
    # Affichage si valeurs au dessus de 2m00
    if f">= {bins[-1]:.2f}m" in counts:
        print(f"         >= {bins[-1]:.2f}m : {counts[f'>= {bins[-1]:.2f}m']} fémur(s)")
        
    print("-" * 40)
    print(f"Total vérifié : {sum(counts.values())} fémurs.")

if __name__ == "__main__":
    main()