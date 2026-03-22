#!/usr/bin/env python
import os
import argparse
import pandas as pd
from tabulate import tabulate

def parse_results(file_path):
    """Extrait les paramètres et résultats du fichier .txt"""
    res = {}
    if not os.path.exists(file_path):
        return None
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if not line: continue

            # --- Parsing des Paramètres ---
            if 'Batch Size :' in line:
                # Gère "Batch Size : 64 | Epochs : 10" ou juste "Batch Size : 64"
                parts = line.split('|')
                res['BS'] = parts[0].split(':')[-1].strip()
                if len(parts) > 1 and 'Epochs' in parts[1]:
                    res['Epochs'] = parts[1].split(':')[-1].strip()
            
            elif 'Epochs :' in line and 'BS' not in res: # Cas où Epochs est seul sur sa ligne
                res['Epochs'] = line.split(':')[-1].strip()

            elif 'Learning Rate :' in line:
                # Extrait 0.001 de "Learning Rate : 0.001 | Optimizer : Adam"
                res['LR'] = line.split(':')[-1].split('|')[0].strip()
            
            elif 'Points :' in line:
                # Gère "Modèle : pointnet2_reg | Points : 2048"
                res['Points'] = line.split(':')[-1].strip()

            # --- Parsing des Résultats (MAE) ---
            # On cherche la valeur numérique avant "cm"
            if 'Best Validation MAE (cm) :' in line:
                val_str = line.split(':')[-1].replace('cm', '').strip()
                res['Val_MAE'] = float(val_str)
            
            # Ton nouveau fichier contient "Final Test MAE (moyenne des erreurs) :" 
            # ou "Final Test MAE (cm) :" selon les versions. On utilise un "in" large :
            elif 'Final Test MAE' in line and 'cm' in line:
                val_str = line.split(':')[-1].replace('cm', '').strip()
                res['Test_MAE'] = float(val_str)
            elif 'Final Test MAE (moyenne des erreurs) :' in line:
                # Si l'unité cm n'est pas écrite mais que c'est la ligne de test
                val_str = line.split(':')[-1].replace('cm', '').strip()
                res['Test_MAE'] = float(val_str)
                
    # Vérification minimale que le fichier n'était pas vide de résultats
    if 'Test_MAE' not in res and 'Val_MAE' not in res:
        return None
        
    return res

def main():
    parser = argparse.ArgumentParser(description='Leaderboard interactif pour Jobs/')
    parser.add_argument('-s', '--sort', default='Test_MAE', help='Champ pour trier (ID, Val_MAE, Test_MAE, BS, LR)')
    parser.add_argument('-r', '--reverse', action='store_true', help='Inverser l\'ordre du tri (par défaut : croissant)')
    parser.add_argument('-csv', action='store_true', help='Exporter en csv')
    args = parser.parse_args()

    # On cherche dans "log/regression" ou "Jobs" selon ton arborescence
    # Si tes dossiers sont dans Jobs/1, Jobs/2...
    jobs_dir = 'Jobs' 
    all_results = []

    if not os.path.exists(jobs_dir):
        print(f"Erreur : Dossier {jobs_dir} introuvable.")
        return

    # Scan des dossiers numériques
    job_ids = [d for d in os.listdir(jobs_dir) if d.isdigit()]
    
    for jid in job_ids:
        result_file = os.path.join(jobs_dir, jid, 'parameters_results.txt')
        data = parse_results(result_file)
        if data:
            data['ID'] = int(jid)
            all_results.append(data)

    if not all_results:
        print("Aucun résultat valide trouvé dans les dossiers de Jobs.")
        return

    df = pd.DataFrame(all_results)
    
    # Réorganisation des colonnes pour la clarté
    cols = ['ID', 'LR', 'BS', 'Epochs', 'Points', 'Val_MAE', 'Test_MAE']
    # On ne garde que les colonnes qui existent réellement dans le DF
    existing_cols = [c for c in cols if c in df.columns]
    df = df[existing_cols]

    # Gestion du tri
    sort_column = args.sort
    mapping = {'id': 'ID', 'mae': 'Test_MAE', 'val': 'Val_MAE', 'lr': 'LR', 'bs': 'BS', 'test': 'Test_MAE'}
    sort_column = mapping.get(sort_column.lower(), sort_column)

    if sort_column in df.columns:
        # Pour les MAE, le meilleur est le plus petit (ascending=True)
        # Sauf si l'utilisateur demande --reverse
        df = df.sort_values(by=sort_column, ascending=not args.reverse)

    # Affichage propre
    print("\n" + "="*95)
    print(f"LEADERBOARD RÉGRESSION FÉMUR (Trié par {sort_column})")
    print("="*95)
    print(tabulate(df, headers='keys', tablefmt='psql', showindex=False))
    
    if args.csv:
        df.to_csv('leaderboard.csv', index=False)
        print("\nSauvegardé dans leaderboard.csv")

if __name__ == "__main__":
    main()