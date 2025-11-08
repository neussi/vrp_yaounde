"""
Module de construction des matrices de couts
Calcule les couts de transport entre chaque paire de stations
pour differentes periodes de la journee
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
import sys
import os
from typing import List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config_global import *


class CostMatrixBuilder:
    """
    Construit les matrices de couts, distances et temps
    pour le probleme de routage
    """
    
    def __init__(self, processed_data: Dict):
        """
        Initialise le constructeur de matrices
        
        Args:
            processed_data: Donnees preprocessees (output de DataPreprocessor)
        """
        self.data = processed_data['data']
        self.station_map = processed_data['station_map']
        self.n_nodes = processed_data['n_stations'] + 1  # +1 pour le depot
        
        # Matrices principales
        self.distance_matrix = None
        self.time_matrix = {}  # {periode: matrice_temps}
        self.cost_matrix = {}  # {periode: matrice_couts}
        
    def build_distance_matrix(self) -> np.ndarray:
        """
        Construit la matrice des distances (km)
        Matrice symmetrique pour simplification
        
        Returns:
            Matrice n x n des distances
        """
        n = self.n_nodes
        dist_matrix = np.zeros((n, n))
        
        # Remplir avec les distances du dataset
        for _, row in self.data.iterrows():
            from_station = row['Station Départ']
            to_station = row['Station Arrivée']
            distance = row['Distance (km)']
            
            if from_station in self.station_map and to_station in self.station_map:
                i = self.station_map[from_station]
                j = self.station_map[to_station]
                
                # Distance depot -> client
                dist_matrix[i, j] = distance
                # Distance client -> depot (symmetrique)
                dist_matrix[j, i] = distance
        
        # Pour les distances inter-clients non specifiees dans le dataset
        # On utilise l'inegalite triangulaire: d(i,j) = d(i,0) + d(0,j)
        for i in range(1, n):
            for j in range(1, n):
                if i != j and dist_matrix[i, j] == 0:
                    dist_matrix[i, j] = dist_matrix[i, 0] + dist_matrix[0, j]
        
        self.distance_matrix = dist_matrix
        print(f"[INFO] Matrice des distances construite: {n}x{n}")
        return dist_matrix
    
    def calculate_travel_time(self, distance: float, coating: str, 
                              traffic: str) -> float:
        """
        Calcule le temps de trajet en fonction de la distance,
        du revetement et du trafic
        
        Args:
            distance: Distance en km
            coating: Type de revetement
            traffic: Niveau de trafic
            
        Returns:
            Temps de trajet en heures
        """
        # Vitesse selon le revetement et le trafic
        if coating in SPEEDS and traffic in SPEEDS[coating]:
            speed = SPEEDS[coating][traffic]
        else:
            # Vitesse par defaut si non trouve
            speed = 30
        
        # Temps = distance / vitesse (en heures)
        travel_time = distance / speed
        return travel_time
    
    def build_time_matrix(self, period: str = 'matin') -> np.ndarray:
        """
        Construit la matrice des temps de trajet pour une periode donnee
        
        Args:
            period: Periode de la journee ('matin', 'midi', 'soir')
            
        Returns:
            Matrice n x n des temps de trajet (heures)
        """
        n = self.n_nodes
        time_matrix = np.zeros((n, n))
        
        # Determiner quelle colonne de trafic utiliser
        if period == 'matin' or period == 'midi':
            traffic_col = 'Trafic Matin'
        else:
            traffic_col = 'Trafic Soir'
        
        # Remplir avec les temps calcules
        for _, row in self.data.iterrows():
            from_station = row['Station Départ']
            to_station = row['Station Arrivée']
            
            if from_station in self.station_map and to_station in self.station_map:
                i = self.station_map[from_station]
                j = self.station_map[to_station]
                
                distance = row['Distance (km)']
                coating = row['Type Revêtement']
                traffic = row[traffic_col]
                
                # Calculer le temps
                travel_time = self.calculate_travel_time(distance, coating, traffic)
                
                # Ajouter le temps de service
                total_time = travel_time + (SERVICE_TIME / 60.0)  # Convertir min en h
                
                time_matrix[i, j] = total_time
                time_matrix[j, i] = total_time  # Symmetrique
        
        # Inter-clients: somme des temps via depot
        for i in range(1, n):
            for j in range(1, n):
                if i != j and time_matrix[i, j] == 0:
                    time_matrix[i, j] = time_matrix[i, 0] + time_matrix[0, j]
        
        self.time_matrix[period] = time_matrix
        print(f"[INFO] Matrice des temps construite pour periode '{period}': {n}x{n}")
        return time_matrix
    
    def calculate_arc_cost(self, row: pd.Series, period: str) -> float:
        """
        Calcule le cout total d'un arc en fonction de tous les facteurs
        
        Args:
            row: Ligne du DataFrame contenant les infos de l'arc
            period: Periode de la journee
            
        Returns:
            Cout total de l'arc (FCFA)
        """
        cost = 0.0
        
        # 1. Cout carburant
        cost += row['Coût Carburant (FCFA)']
        
        # 2. Frais informels
        cost += row['Frais Informels (FCFA)']
        
        # 3. Peages
        cost += row['Péages (FCFA)']
        
        # 4. Penalite selon stationnement
        parking = row['Stationnement']
        if parking == 'Très difficile':
            cost += PENALTY_PARKING * 3
        elif parking in ['Difficile', 'Limité']:
            cost += PENALTY_PARKING * 2
        elif parking == 'Très limité':
            cost += PENALTY_PARKING * 4
        
        # 5. Facteur de congestion selon periode
        if period in ['matin', 'soir']:
            traffic_col = 'Trafic Matin' if period == 'matin' else 'Trafic Soir'
            traffic = row[traffic_col]
            
            if traffic == 'Très dense':
                cost *= 1.5  # Augmentation de 50% en congestion severe
            elif traffic == 'Dense':
                cost *= 1.3
        
        # 6. Penalite selon etat de la route
        road_state = row['État Route']
        if road_state == 'Mauvais':
            cost += 500  # Cout additionnel pour usure vehicule
        
        return cost
    
    def build_cost_matrix(self, period: str = 'matin') -> np.ndarray:
        """
        Construit la matrice des couts pour une periode donnee
        
        Args:
            period: Periode de la journee ('matin', 'midi', 'soir')
            
        Returns:
            Matrice n x n des couts (FCFA)
        """
        n = self.n_nodes
        cost_matrix = np.zeros((n, n))
        
        # Remplir avec les couts calcules
        for _, row in self.data.iterrows():
            from_station = row['Station Départ']
            to_station = row['Station Arrivée']
            
            if from_station in self.station_map and to_station in self.station_map:
                i = self.station_map[from_station]
                j = self.station_map[to_station]
                
                # Calculer le cout
                arc_cost = self.calculate_arc_cost(row, period)
                
                cost_matrix[i, j] = arc_cost
                cost_matrix[j, i] = arc_cost  # Symmetrique
        
        # Inter-clients: somme des couts via depot
        for i in range(1, n):
            for j in range(1, n):
                if i != j and cost_matrix[i, j] == 0:
                    cost_matrix[i, j] = cost_matrix[i, 0] + cost_matrix[0, j]
        
        self.cost_matrix[period] = cost_matrix
        print(f"[INFO] Matrice des couts construite pour periode '{period}': {n}x{n}")
        return cost_matrix
    
    def build_all_matrices(self) -> Dict[str, np.ndarray]:
        """
        Construit toutes les matrices pour toutes les periodes
        
        Returns:
            Dictionnaire contenant toutes les matrices
        """
        print("\n" + "="*60)
        print("CONSTRUCTION DES MATRICES")
        print("="*60 + "\n")
        
        # Matrice des distances (independante du temps)
        self.build_distance_matrix()
        
        # Matrices temps et couts pour chaque periode
        periods = ['matin', 'midi', 'soir']
        for period in periods:
            self.build_time_matrix(period)
            self.build_cost_matrix(period)
        
        matrices = {
            'distance': self.distance_matrix,
            'time': self.time_matrix,
            'cost': self.cost_matrix,
            'n_nodes': self.n_nodes
        }
        
        print("\n" + "="*60)
        print("CONSTRUCTION DES MATRICES TERMINEE")
        print("="*60 + "\n")
        
        return matrices
    
    def get_statistics(self) -> Dict:
        """
        Calcule des statistiques sur les matrices
        
        Returns:
            Dictionnaire de statistiques
        """
        stats = {}
        
        if self.distance_matrix is not None:
            stats['distance'] = {
                'min': np.min(self.distance_matrix[self.distance_matrix > 0]),
                'max': np.max(self.distance_matrix),
                'mean': np.mean(self.distance_matrix[self.distance_matrix > 0])
            }
        
        if self.cost_matrix:
            for period, matrix in self.cost_matrix.items():
                stats[f'cost_{period}'] = {
                    'min': np.min(matrix[matrix > 0]),
                    'max': np.max(matrix),
                    'mean': np.mean(matrix[matrix > 0])
                }
        
        return stats


def main():
    """
    Fonction principale pour tester la construction des matrices
    """
    import pickle
    
    # Charger les donnees preprocessees
    data_path = os.path.join(DATA_DIR, 'processed_data.pkl')
    with open(data_path, 'rb') as f:
        processed_data = pickle.load(f)
    
    # Construire les matrices
    builder = CostMatrixBuilder(processed_data)
    matrices = builder.build_all_matrices()
    
    # Afficher statistiques
    stats = builder.get_statistics()
    print("\n--- STATISTIQUES DES MATRICES ---")
    for key, values in stats.items():
        print(f"\n{key.upper()}:")
        for stat, val in values.items():
            print(f"  {stat}: {val:.2f}")
    
    # Sauvegarder les matrices
    output_path = os.path.join(DATA_DIR, 'matrices.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(matrices, f)
    print(f"\n[INFO] Matrices sauvegardees: {output_path}")


if __name__ == "__main__":
    main()