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

    # 3. Filtrer pour ne garder que les vrais fémurs de base (sans aucune augmentation ACVD _aug)
    all_real_meshes = [
        entry for rid in unique_root_ids 
        for entry in groups[rid] 
        if '_aug' not in entry['obj_path']
    ]

    print(f"==========================================")
    print(f"===     STATS ON THE ENTIRE DATASET      ===")
    print(f"===          (Non-augmented)           ===")
    print(f"==========================================")
    print(f"Total number of unique patients: {len(unique_root_ids)}")
    print(f"Total number of real meshes    : {len(all_real_meshes)}\n")

    # 4. Extraction des tailles par Genre
    male_sizes = []
    female_sizes = []
    gender_counts = defaultdict(int)
    gender_key = "PatientSex"

    for item in all_real_meshes:
        size = item['PatientSize']
        gender_val = str(item[gender_key]).strip().upper()
        
        # Standardisation vers labels anglais et tri des tailles
        if gender_val in ["M", "H", "MALE", "HOMME"]:
            gender_counts['Males (M)'] += 1
            male_sizes.append(size)
        elif gender_val in ["F", "FEMALE", "FEMME"]:
            gender_counts['Females (F)'] += 1
            female_sizes.append(size)
        else:
            gender_counts[f'Other ({gender_val})'] += 1

    # 5. Définition des tranches de taille de 5cm en 5cm (en mètres)
    bins = np.arange(1.45, 2.05, 0.05)
    
    # Préparation des labels d'intervalles standards
    intervals = []
    for i in range(len(bins) - 1):
        intervals.append(f"{bins[i]:.2f}m - {bins[i+1]:.2f}m")
    
    # Initialisation des compteurs ordonnés par genre
    counts_males = {interv: 0 for interv in intervals}
    counts_females = {interv: 0 for interv in intervals}
    
    # Remplissage pour les Hommes
    for size in male_sizes:
        for i in range(len(bins) - 1):
            if bins[i] <= size < bins[i+1]:
                counts_males[f"{bins[i]:.2f}m - {bins[i+1]:.2f}m"] += 1
                break

    # Remplissage pour les Femmes
    for size in female_sizes:
        for i in range(len(bins) - 1):
            if bins[i] <= size < bins[i+1]:
                counts_females[f"{bins[i]:.2f}m - {bins[i+1]:.2f}m"] += 1
                break

    # 6. Affichage Textuel de contrôle (Terminal)
    print("Gender Distribution:")
    print("-" * 40)
    # Assurer l'affichage ordonné : Hommes puis Femmes
    for g_key in ['Males (M)', 'Females (F)']:
        if g_key in gender_counts:
            count = gender_counts[g_key]
            percentage = (count / len(all_real_meshes)) * 100 if all_real_meshes else 0
            print(f"  {g_key:<20} : {count:>3} femur(s) ({percentage:.1f}%)")
    print()

    print("Height Distribution (Males vs Females):")
    print("-" * 55)
    for interv in intervals:
        print(f"  {interv:<15} | Males: {counts_males[interv]:>2} | Females: {counts_females[interv]:>2}")
    print("-" * 55)
    print(f"Total verified: {len(male_sizes) + len(female_sizes)} femurs.\n")

    # ==========================================================
    # 7. GENERATION DES GRAPHIQUES
    # ==========================================================
    print("-> Generating charts...")
    
    # Style global épuré
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # --- CHART 1 : Gender Distribution (Pie Chart) ---
    plt.figure(figsize=(6, 6))
    # Correction de l'ordre pour matcher avec l'ordre des couleurs (Bleu = Homme, Rose = Femme)
    labels = ["Males (M)", "Females (F)"]
    sizes = [gender_counts['Males (M)'], gender_counts['Females (F)']]
    
    colors = ['#5dade2', '#f1948a'] 
    
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, 
            colors=colors, wedgeprops={'edgecolor': 'white', 'linewidth': 2})
    plt.tight_layout()
    
    plot_gender_path = "gender_distribution.png"
    plt.savefig(plot_gender_path, dpi=300)
    print(f"   [OK] Gender chart saved to: {plot_gender_path}")
    plt.close()

    # --- CHART 2 : Height Distribution (Grouped Bar Chart) ---
    plt.figure(figsize=(12, 6))
    
    x = np.arange(len(intervals))  # Emplacement des groupes sur l'axe X
    width = 0.35  # Largeur des barres
    
    male_values = [counts_males[interv] for interv in intervals]
    female_values = [counts_females[interv] for interv in intervals]
    
    # Génération des deux barres côte à côte par intervalle
    bars_m = plt.bar(x - width/2, male_values, width, label='Males (M)', color='#5dade2', edgecolor='#2980b9', alpha=0.9)
    bars_f = plt.bar(x + width/2, female_values, width, label='Females (F)', color='#f1948a', edgecolor='#c0392b', alpha=0.9)
    
    # Ajouter les totaux au-dessus de chaque barre (Hommes)
    for bar in bars_m:
        yval = bar.get_height()
        if yval > 0:
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.3, str(yval), ha='center', va='bottom', fontsize=9, fontweight='bold', color='#2c3e50')

    # Ajouter les totaux au-dessus de chaque barre (Femmes)
    for bar in bars_f:
        yval = bar.get_height()
        if yval > 0:
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.3, str(yval), ha='center', va='bottom', fontsize=9, fontweight='bold', color='#2c3e50')

    # Ajustements graphiques
    plt.xlabel("Stature Intervals (m)", fontsize=12, labelpad=10)
    plt.ylabel("Number of Femurs", fontsize=12, labelpad=10)
    plt.xticks(x, intervals, rotation=25, ha='right')
    
    max_val = max(max(male_values), max(female_values))
    plt.ylim(0, max_val + max_val * 0.15)
    
    plt.legend(fontsize=11, loc='upper right')
    plt.tight_layout()
    
    plot_sizes_path = "height_distribution_by_gender.png"
    plt.savefig(plot_sizes_path, dpi=300)
    print(f"   [OK] Height by gender chart saved to: {plot_sizes_path}")
    plt.close()
    
    print("\nAll charts have been customized and successfully saved!")

if __name__ == "__main__":
    main()