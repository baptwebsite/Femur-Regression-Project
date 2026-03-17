#!/bin/bash

# Listes des paramètres à tester
LRS=(0.001)
BSS=(16 32 64)
NPTS=(1024 2048)

# LRS=(0.001 0.0001 0.00001)
# BSS=(16 32 64 128 256)
# NPTS=(1024 2048 4096)

# METTRE À "true" POUR SIMULER, "false" POUR LANCER VRAIMENT
SIMULATION=false

# --- BOUCLES ---
for lr in "${LRS[@]}"; do
    for bs in "${BSS[@]}"; do
        for np in "${NPTS[@]}"; do
            
            LOG_NAME="LR${lr}_BS${bs}_NP${np}"
            
            # On construit la commande dans une variable
            CMD="python submit.py -c config.yaml -p learning_rate $lr -p batch_size $bs -p num_point $np -p epoch 1000 -g"

            if [ "$SIMULATION" = true ] ; then
                echo "[SIMULATION] $CMD"
            else
                echo "[EXECUTION] Lancement de $LOG_NAME..."
                $CMD  # Ici la commande est réellement exécutée
                sleep 1
            fi

        done
    done
done

if [ "$SIMULATION" = true ] ; then
    echo -e "\n--- FIN DE LA SIMULATION ---"
    echo "Si les commandes ci-dessus sont correctes, change SIMULATION=false dans le script."
fi