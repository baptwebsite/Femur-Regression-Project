#!/usr/bin/env python
import os
import argparse
import pandas as pd
from tabulate import tabulate

def parse_results(file_path):
    """Extrait proprement les paramètres et résultats du fichier text"""
    res = {}
    if not os.path.exists(file_path):
        return None
    
    # Valeurs par défaut pour les anciens jobs qui n'avaient pas ces paramètres
    res['Sampling'] = 'N/A'
    res['Augment'] = 'N/A'
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line: continue

            # 1. Extraction des Paramètres (LR, BS, Epochs, Points)
            if 'Learning Rate' in line:
                res['LR'] = line.split(':')[-1].split('|')[0].strip()
            
            if 'Batch Size' in line:
                parts = line.split('|')
                res['BS'] = parts[0].split(':')[-1].strip()
                for p in parts:
                    if 'Epochs' in p:
                        res['Epochs'] = p.split(':')[-1].strip()

            if 'Points demandés' in line or 'Points' in line:
                res['Points'] = line.split(':')[-1].strip()
                
            if "Méthode d'échantillonnage" in line:
                res['Sampling'] = line.split(':')[-1].strip().upper()
                
            if "Utilisation Augmentation" in line:
                res['Augment'] = line.split(':')[-1].strip()
            
            # Cas de secours si Epochs est sur sa propre ligne
            if 'Epochs :' in line and 'Epochs' not in res:
                res['Epochs'] = line.split(':')[-1].strip()

            # 2. Extraction des MAE (cm)
            if 'Best Validation MAE (cm)' in line:
                val_str = line.split(':')[-1].replace('cm', '').strip()
                res['Val_MAE'] = float(val_str)
            
            # Gestion flexible du nom pour le Test MAE
            if 'Final Test MAE' in line:
                test_str = line.split(':')[-1].replace('cm', '').strip()
                test_str = test_str.split('(')[0].strip()
                try:
                    res['Test_MAE'] = float(test_str)
                except ValueError:
                    continue
                
    return res if ('Test_MAE' in res or 'Val_MAE' in res) else None

def main():
    parser = argparse.ArgumentParser(description='Leaderboard interactif pour Jobs/')
    parser.add_argument('-s', '--sort', default='Test_MAE', 
                        help='Champ pour trier (ID, Val_MAE, Test_MAE, BS, LR, Epochs, Sampling, Augment)')
    parser.add_argument('-r', '--reverse', action='store_true', help='Inverser l\'ordre du tri')
    parser.add_argument('-csv', action='store_true', help='Exporter en csv')
    args = parser.parse_args()

    jobs_dir = 'Jobs'
    all_results = []

    if not os.path.exists(jobs_dir):
        print(f"Erreur : Dossier {jobs_dir} introuvable.")
        return

    # Scan des dossiers numériques dans Jobs/
    job_ids = sorted([d for d in os.listdir(jobs_dir) if d.isdigit()], key=int)
    
    for jid in job_ids:
        result_file = os.path.join(jobs_dir, jid, 'parameters_results.txt')
        data = parse_results(result_file)
        if data:
            data['ID'] = int(jid)
            all_results.append(data)

    if not all_results:
        print("Aucun résultat exploitable trouvé (fichiers parameters_results.txt absents ou vides).")
        return

    df = pd.DataFrame(all_results)
    
    # --- RÉORGANISATION ET TRI ---
    desired_cols = ['ID', 'LR', 'BS', 'Epochs', 'Points', 'Sampling', 'Augment', 'Val_MAE', 'Test_MAE']
    cols = [c for c in desired_cols if c in df.columns]
    df = df[cols]

    # Remplacement des valeurs vides résiduelles (au cas où) par une chaîne de caractères
    df = df.fillna('N/A')
    df['Augment'] = df['Augment'].astype(str)
    df['Sampling'] = df['Sampling'].astype(str)

    # Gestion du tri
    sort_column = args.sort
    mapping = {
        'id': 'ID', 'mae': 'Test_MAE', 'test': 'Test_MAE', 'val': 'Val_MAE', 
        'lr': 'LR', 'bs': 'BS', 'ep': 'Epochs', 'samp': 'Sampling', 'aug': 'Augment'
    }
    sort_column = mapping.get(sort_column.lower(), sort_column)

    if sort_column in df.columns:
        is_metric = 'MAE' in sort_column
        df = df.sort_values(by=sort_column, ascending=is_metric if not args.reverse else not is_metric)

    # Affichage de la table
    print("\n" + "="*110)
    print(f"LEADERBOARD RÉGRESSION FÉMUR (Trié par {sort_column})")
    print("="*110)
    print(tabulate(df, headers='keys', tablefmt='psql', showindex=False))
    
    if args.csv:
        df.to_csv('leaderboard.csv', index=False)
        print("\nSauvegardé dans leaderboard.csv")

if __name__ == "__main__":
    main()