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
    
    with open(file_path, 'r') as f:
        content = f.read()
        lines = content.split('\n')
        
        for line in lines:
            # Extraction Batch Size et Epochs (Gestion de la ligne avec '|')
            if 'Batch Size' in line:
                parts = line.split('|')
                res['BS'] = parts[0].split(':')[-1].strip()
                if len(parts) > 1 and 'Epochs' in parts[1]:
                    res['Epochs'] = parts[1].split(':')[-1].strip()
            
            # Extraction Learning Rate
            if 'Learning Rate' in line:
                res['LR'] = line.split(':')[-1].split('|')[0].strip()
            
            # Extraction Points
            if 'Points :' in line:
                res['Points'] = line.split(':')[-1].strip()

            # Extraction des MAE (cm)
            if 'Best Validation MAE (cm)' in line:
                res['Val_MAE'] = float(line.split(':')[-1].replace('cm', '').strip())
            if 'Final Test MAE (cm)' in line:
                res['Test_MAE'] = float(line.split(':')[-1].replace('cm', '').strip())
                
    return res

def main():
    parser = argparse.ArgumentParser(description='Leaderboard interactif pour Jobs/')
    parser.add_argument('-s', '--sort', default='Test_MAE', help='Champ pour trier (ID, Val_MAE, Test_MAE, BS, LR)')
    parser.add_argument('-r', '--reverse', action='store_true', help='Inverser l\'ordre du tri')
    parser.add_argument('-csv', action='store_true', help='Exporter en csv')
    args = parser.parse_args()

    jobs_dir = 'Jobs'
    all_results = []

    if not os.path.exists(jobs_dir):
        print(f"Erreur : Dossier {jobs_dir} introuvable.")
        return

    # Scan des dossiers numériques dans Jobs/
    job_ids = [d for d in os.listdir(jobs_dir) if d.isdigit()]
    
    for jid in job_ids:
        result_file = os.path.join(jobs_dir, jid, 'parameters_results.txt')
        data = parse_results(result_file)
        if data:
            data['ID'] = int(jid)
            all_results.append(data)

    if not all_results:
        print("Aucun résultat trouvé.")
        return

    df = pd.DataFrame(all_results)
    
    # Réorganisation des colonnes
    cols = ['ID', 'LR', 'BS', 'Epochs', 'Points', 'Val_MAE', 'Test_MAE']
    df = df[cols]

    # Gestion du tri via les arguments du terminal
    sort_column = args.sort
    # Map pour tolérer les minuscules ou noms courts
    mapping = {'id': 'ID', 'mae': 'Test_MAE', 'val': 'Val_MAE', 'lr': 'LR', 'bs': 'BS'}
    sort_column = mapping.get(sort_column.lower(), sort_column)

    if sort_column in df.columns:
        df = df.sort_values(by=sort_column, ascending=not args.reverse)

    # Affichage
    print("\n" + "="*85)
    print(f"LEADERBOARD RÉGRESSION FÉMUR (Trié par {sort_column})")
    print("="*85)
    print(tabulate(df, headers='keys', tablefmt='psql', showindex=False))
    
    if args.csv:
        df.to_csv('leaderboard.csv', index=False)
        print("\nSauvegardé dans leaderboard.csv")

if __name__ == "__main__":
    main()