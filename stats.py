import os
import json
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

def main():
    # CONFIGURATION DES CHEMINS
    root_dataset = "data/" 
    json_path = os.path.join(root_dataset, 'full_dataset.json')
    
    if not os.path.exists(json_path):
        print(f"Error: The file {json_path} could not be found.")
        return

    # 1. Charger le JSON d'indexation
    with open(json_path, 'r') as f:
        all_data = json.load(f)

    # 2. Regrouper par Patient pour compter les individus uniques
    groups = defaultdict(list)
    for item in all_data:
        filename = os.path.basename(item['obj_path'])
        root_id = filename.split('_aug')[0].replace('.obj', '')
        groups[root_id].append(item)
    
    unique_root_ids = sorted(list(groups.keys()))

    # 3. Filtrer pour ne garder que les vrais fémurs de base (sans aucune augmentation _aug)
    all_real_meshes = [
        entry for rid in unique_root_ids 
        for entry in groups[rid] 
        if '_aug' not in entry['obj_path']
    ]

    print(f"==========================================")
    print(f"===    STATS ON THE ENTIRE DATASET     ===")
    print(f"===          (Non-augmented)           ===")
    print(f"==========================================")
    print(f"Total number of unique patients: {len(unique_root_ids)}")
    print(f"Total number of real meshes    : {len(all_real_meshes)}\n")

    # 4. Extraction des tailles et du genre (avec labels anglais)
    true_sizes = []
    gender_counts = defaultdict(int)
    gender_key = "PatientSex"

    for item in all_real_meshes:
        true_sizes.append(item['PatientSize'])
        
        gender_val = str(item[gender_key]).strip().upper()
        # Standardisation vers labels anglais (Males / Females)
        if gender_val in ["M", "H", "MALE", "HOMME"]:
            gender_counts['Males (M)'] += 1
        elif gender_val in ["F", "FEMALE", "FEMME"]:
            gender_counts['Femmes (F)'] += 1
        else:
            gender_counts[f'Other ({gender_val})'] += 1

    # 5. Définition des tranches de taille de 5cm en 5cm (en mètres)
    bins = np.arange(1.45, 2.05, 0.05)
    
    # Préparation des labels d'intervalles standards
    intervals = []
    for i in range(len(bins) - 1):
        intervals.append(f"{bins[i]:.2f}m - {bins[i+1]:.2f}m")
    
    # Initialisation des compteurs ordonnés pour le graphique
    counts_dict = {interv: 0 for interv in intervals}
    counts_dict["< 1.45m"] = 0
    counts_dict[">= 2.00m"] = 0
    
    for size in true_sizes:
        found = False
        for i in range(len(bins) - 1):
            low = bins[i]
            high = bins[i+1]
            if low <= size < high:
                counts_dict[f"{low:.2f}m - {high:.2f}m"] += 1
                found = True
                break
        if not found:
            if size < bins[0]:
                counts_dict["< 1.45m"] += 1
            elif size >= bins[-1]:
                counts_dict[">= 2.00m"] += 0

    # 6. Affichage Textuel de contrôle (Terminal)
    print("Gender Distribution:")
    print("-" * 40)
    for gender, count in gender_counts.items():
        percentage = (count / len(all_real_meshes)) * 100 if all_real_meshes else 0
        print(f"  {gender:<20} : {count:>3} femur(s) ({percentage:.1f}%)")
    print()

    print("Height Distribution:")
    print("-" * 40)
    if counts_dict["< 1.45m"] > 0:
        print(f"          < 1.45m : {counts_dict['< 1.45m']} femur(s)")
    for interv in intervals:
        print(f"  Between {interv} : {counts_dict[interv]} femur(s)")
    if counts_dict[">= 2.00m"] > 0:
        print(f"         >= 2.00m : {counts_dict['>= 2.00m']} femur(s)")
    print("-" * 40)
    print(f"Total verified: {sum(counts_dict.values())} femurs.\n")

    # ==========================================================
    # 7. GENERATION DES GRAPHIQUES TRADUITS
    # ==========================================================
    print("-> Generating charts...")
    
    # Style global épuré
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # --- CHART 1 : Gender Distribution (Pie Chart) ---
    plt.figure(figsize=(6, 6))
    labels = ["Females (F)", "Males (M)"]
    sizes = list(gender_counts.values())
    
    # COULEURS CLAIRES : Bleu ciel (#5dade2), Rouge/Rose corail (#f1948a), Gris (#bdc3c7)
    colors = ['#5dade2', '#f1948a', '#bdc3c7'] 
    
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, 
            colors=colors[:len(labels)], wedgeprops={'edgecolor': 'white', 'linewidth': 2})
    # plt.title("Gender Distribution in the Global Dataset", fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    plot_gender_path = "gender_distribution.png"
    plt.savefig(plot_gender_path, dpi=300)
    print(f"   [OK] Gender chart saved to: {plot_gender_path}")
    plt.close()

    # --- CHART 2 : Height Distribution (Bar Chart) ---
    plt.figure(figsize=(10, 6))
    
    # Ordonner les données chronologiquement de gauche à droite
    ordered_labels = []
    if counts_dict["< 1.45m"] > 0: ordered_labels.append("< 1.45m")
    ordered_labels.extend(intervals)
    if counts_dict[">= 2.00m"] > 0: ordered_labels.append(">= 2.00m")
    
    ordered_values = [counts_dict[lbl] for lbl in ordered_labels]
    
    # COULEUR DES BARRES : Bleu clair (#85c1e9)
    bars = plt.bar(ordered_labels, ordered_values, color='#85c1e9', edgecolor='#5499c7', alpha=0.9, width=0.6)
    
    # Ajouter les totaux au-dessus de chaque barre
    for bar in bars:
        yval = bar.get_height()
        if yval > 0:
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, str(yval), ha='center', va='bottom', fontweight='bold', color='#2c3e50')

    # plt.title("Frequency Distribution of Patient Heights", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Stature Intervals (m)", fontsize=12, labelpad=10)
    plt.ylabel("Number of Femurs", fontsize=12, labelpad=10)
    plt.xticks(rotation=25, ha='right')
    plt.ylim(0, max(ordered_values) + (max(ordered_values) * 0.15))
    plt.tight_layout()
    
    plot_sizes_path = "height_distribution.png"
    plt.savefig(plot_sizes_path, dpi=300)
    print(f"   [OK] Height chart saved to: {plot_sizes_path}")
    plt.close()
    
    print("\nAll charts have been translated to English and successfully saved!")

if __name__ == "__main__":
    main()