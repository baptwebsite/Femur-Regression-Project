#!/bin/bash

# Listes des paramètres à tester
LRS=(0.001 0.0001)
BSS=(64 128 256)
NPTS=(2048 4096)

# METTRE À "true" POUR SIMULER, "false" POUR LANCER VRAIMENT
SIMULATION=false

for lr in "${LRS[@]}"; do
    for bs in "${BSS[@]}"; do
        for np in "${NPTS[@]}"; do
            
            LOG_NAME="LR${lr}_BS${bs}_NP${np}"
            
            # Définition des arguments pour submit.py
            # On passe les paramètres un par un pour que Bash gère correctement les espaces
            ARGS=(-c config.yaml -p learning_rate "$lr" -p batch_size "$bs" -p num_point "$np" -p epoch 300)
            
            # Si ce n'est pas une simulation, on ajoute le flag de soumission effective (-g ou --go)
            if [ "$SIMULATION" = false ] ; then
                ARGS+=("-g")
            fi

            if [ "$SIMULATION" = true ] ; then
                echo "[SIMULATION] python submit.py ${ARGS[*]}"
            else
                echo "[EXECUTION] Lancement de $LOG_NAME via submit.py..."
                python submit.py "${ARGS[@]}"
                sleep 1 # Pause de sécurité entre chaque soumission Slurm
            fi

        done
    done
done

if [ "$SIMULATION" = true ] ; then
    echo -e "\n--- FIN DE LA SIMULATION ---"
    echo "Si les commandes ci-dessus sont correctes, change SIMULATION=false dans le script."
fi