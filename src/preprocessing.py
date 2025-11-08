"""
Module de prétraitement des données du VRP Yaoundé

Fonctions:
- Chargement et validation des données CSV
- Nettoyage et transformation
- Gestion des valeurs manquantes
- Normalisation des formats
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

class DataPreprocessor:
    """
    Classe pour le prétraitement des données du VRP
    
    Gère le chargement, nettoyage et transformation des données
    avant leur utilisation dans l'algorithme d'optimisation.
    """
    
    def __init__(self, data=None):
        """
        Initialise le preprocesseur
        
        Args:
            data: DataFrame pandas ou chemin vers fichier CSV (optionnel)
        """
        if isinstance(data, pd.DataFrame):
            # Si on reçoit directement un DataFrame
            self.data = data.copy()
        elif isinstance(data, str):
            # Si on reçoit un chemin de fichier
            self.data = pd.read_csv(data)
        else:
            # Pas de données
            self.data = None
        
        self.cleaned_data = None
        self.station_mapping = {}  # Mapping station_name -> node_id
        self.node_mapping = {}     # Mapping node_id -> station_name
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Charge les données depuis un fichier CSV
        
        Args:
            filepath: Chemin vers le fichier CSV
            
        Returns:
            DataFrame pandas avec les données chargées
        """
        print(f"Chargement des donnees depuis: {filepath}")
        self.data = pd.read_csv(filepath)
        print(f"Donnees chargees: {len(self.data)} lignes, {len(self.data.columns)} colonnes")
        return self.data
    
    def clean_data(self) -> pd.DataFrame:
        """
        Nettoie et transforme les données
        
        Opérations:
        - Suppression des doublons
        - Gestion des valeurs manquantes
        - Normalisation des formats
        - Validation des types
        
        Returns:
            DataFrame nettoyé
        """
        if self.data is None:
            raise ValueError("Aucune donnee chargee. Fournissez un DataFrame ou appelez load_data()")
        
        print("\nNettoyage des donnees...")
        df = self.data.copy()
        
        # Afficher les colonnes disponibles
        print(f"Colonnes disponibles: {list(df.columns)}")
        
        # 1. Supprimer les doublons
        initial_rows = len(df)
        df = df.drop_duplicates()
        duplicates_removed = initial_rows - len(df)
        if duplicates_removed > 0:
            print(f"  - Doublons supprimes: {duplicates_removed}")
        
        # 2. Nettoyer les noms de colonnes (enlever espaces)
        df.columns = df.columns.str.strip()
        
        # 3. Gérer les valeurs manquantes dans les colonnes numériques
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if df[col].isnull().any():
                # Remplacer par la médiane
                median_value = df[col].median()
                df[col].fillna(median_value, inplace=True)
                print(f"  - Valeurs manquantes dans '{col}' remplacees par mediane: {median_value:.2f}")
        
        # 4. Nettoyer les colonnes textuelles
        text_columns = df.select_dtypes(include=['object']).columns
        for col in text_columns:
            if df[col].isnull().any():
                # Remplacer par 'Inconnu' ou valeur par défaut
                df[col].fillna('Inconnu', inplace=True)
                print(f"  - Valeurs manquantes dans '{col}' remplacees par 'Inconnu'")
            
            # Nettoyer les espaces
            df[col] = df[col].str.strip()
        
        # 5. Valider les colonnes essentielles
        required_columns = ['Station Départ', 'Station Arrivée', 'Distance (km)']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"ATTENTION: Colonnes manquantes: {missing_columns}")
        
        # 6. Créer le mapping des stations vers IDs numériques
        self._create_station_mapping(df)
        
        # 7. Valider les distances (doivent être positives)
        if 'Distance (km)' in df.columns:
            negative_distances = df['Distance (km)'] < 0
            if negative_distances.any():
                print(f"  - ATTENTION: {negative_distances.sum()} distances negatives detectees")
                df.loc[negative_distances, 'Distance (km)'] = df['Distance (km)'].abs()
        
        # 8. Ajouter une colonne ID si elle n'existe pas
        if 'ID' not in df.columns:
            df.insert(0, 'ID', range(1, len(df) + 1))
        
        self.cleaned_data = df
        print(f"\nNettoyage termine: {len(df)} lignes conservees")
        
        return df
    
    def _create_station_mapping(self, df: pd.DataFrame):
        """
        Crée un mapping entre noms de stations et IDs numériques
        
        Le dépôt central aura toujours l'ID 0
        Les autres stations auront des IDs séquentiels
        
        Args:
            df: DataFrame avec les données
        """
        # Récupérer toutes les stations uniques
        all_stations = set()
        
        if 'Station Départ' in df.columns:
            all_stations.update(df['Station Départ'].unique())
        if 'Station Arrivée' in df.columns:
            all_stations.update(df['Station Arrivée'].unique())
        
        # Le dépôt central est toujours ID 0
        depot_names = ['Dépôt Central', 'Depot Central', 'Dépôt Central (Mvan)']
        depot = None
        
        for depot_name in depot_names:
            if depot_name in all_stations:
                depot = depot_name
                break
        
        if depot is None:
            # Prendre la première station comme dépôt
            depot = list(all_stations)[0]
            print(f"  - ATTENTION: Depot non trouve, utilisation de '{depot}' comme depot")
        
        # Créer le mapping
        self.station_mapping[depot] = 0
        self.node_mapping[0] = depot
        
        # Attribuer des IDs aux autres stations
        node_id = 1
        for station in sorted(all_stations):
            if station != depot and station not in self.station_mapping:
                self.station_mapping[station] = node_id
                self.node_mapping[node_id] = station
                node_id += 1
        
        print(f"  - Mapping cree: {len(self.station_mapping)} stations")
        print(f"    Depot: {depot} (ID=0)")
        print(f"    Clients: {len(self.station_mapping)-1}")
    
    def get_station_id(self, station_name: str) -> int:
        """
        Retourne l'ID numérique d'une station
        
        Args:
            station_name: Nom de la station
            
        Returns:
            ID numérique (0 pour dépôt, 1+ pour clients)
        """
        return self.station_mapping.get(station_name, -1)
    
    def get_station_name(self, node_id: int) -> str:
        """
        Retourne le nom d'une station depuis son ID
        
        Args:
            node_id: ID numérique
            
        Returns:
            Nom de la station
        """
        return self.node_mapping.get(node_id, "Inconnu")
    
    def get_summary_statistics(self) -> Dict:
        """
        Calcule des statistiques descriptives sur les données
        
        Returns:
            Dictionnaire avec statistiques
        """
        if self.cleaned_data is None:
            raise ValueError("Aucune donnee nettoyee disponible")
        
        df = self.cleaned_data
        stats = {
            'num_rows': len(df),
            'num_stations': len(self.station_mapping),
            'num_clients': len(self.station_mapping) - 1,  # Exclure dépôt
        }
        
        # Statistiques sur les distances
        if 'Distance (km)' in df.columns:
            stats['distance_stats'] = {
                'mean': df['Distance (km)'].mean(),
                'median': df['Distance (km)'].median(),
                'min': df['Distance (km)'].min(),
                'max': df['Distance (km)'].max(),
                'total': df['Distance (km)'].sum()
            }
        
        # Statistiques sur les coûts
        if 'Coût Carburant (FCFA)' in df.columns:
            stats['cost_stats'] = {
                'mean': df['Coût Carburant (FCFA)'].mean(),
                'median': df['Coût Carburant (FCFA)'].median(),
                'min': df['Coût Carburant (FCFA)'].min(),
                'max': df['Coût Carburant (FCFA)'].max(),
                'total': df['Coût Carburant (FCFA)'].sum()
            }
        
        return stats
    
    def print_summary(self):
        """Affiche un résumé des données"""
        if self.cleaned_data is None:
            print("Aucune donnee nettoyee disponible")
            return
        
        stats = self.get_summary_statistics()
        
        print("\n" + "="*60)
        print("RESUME DES DONNEES")
        print("="*60)
        print(f"Nombre de trajets: {stats['num_rows']}")
        print(f"Nombre de stations: {stats['num_stations']}")
        print(f"Nombre de clients: {stats['num_clients']}")
        
        if 'distance_stats' in stats:
            print(f"\nStatistiques distances (km):")
            print(f"  - Moyenne: {stats['distance_stats']['mean']:.2f}")
            print(f"  - Mediane: {stats['distance_stats']['median']:.2f}")
            print(f"  - Min: {stats['distance_stats']['min']:.2f}")
            print(f"  - Max: {stats['distance_stats']['max']:.2f}")
            print(f"  - Total: {stats['distance_stats']['total']:.2f}")
        
        if 'cost_stats' in stats:
            print(f"\nStatistiques couts carburant (FCFA):")
            print(f"  - Moyenne: {stats['cost_stats']['mean']:.0f}")
            print(f"  - Mediane: {stats['cost_stats']['median']:.0f}")
            print(f"  - Min: {stats['cost_stats']['min']:.0f}")
            print(f"  - Max: {stats['cost_stats']['max']:.0f}")
            print(f"  - Total: {stats['cost_stats']['total']:.0f}")
        
        print("="*60 + "\n")
    
    def validate_data_quality(self) -> Tuple[bool, List[str]]:
        """
        Valide la qualité des données
        
        Returns:
            (est_valide, liste_problemes)
        """
        if self.cleaned_data is None:
            return False, ["Aucune donnee nettoyee disponible"]
        
        df = self.cleaned_data
        problems = []
        
        # Vérifier les colonnes essentielles
        required_cols = ['Station Départ', 'Station Arrivée', 'Distance (km)']
        for col in required_cols:
            if col not in df.columns:
                problems.append(f"Colonne manquante: {col}")
        
        # Vérifier les valeurs nulles
        null_counts = df.isnull().sum()
        for col, count in null_counts.items():
            if count > 0:
                problems.append(f"Colonne '{col}' contient {count} valeurs nulles")
        
        # Vérifier les distances négatives
        if 'Distance (km)' in df.columns:
            if (df['Distance (km)'] < 0).any():
                problems.append("Distances negatives detectees")
        
        # Vérifier qu'il y a au moins un dépôt
        if len(self.station_mapping) == 0:
            problems.append("Aucune station mappee")
        
        is_valid = len(problems) == 0
        return is_valid, problems
    
    def export_cleaned_data(self, filepath: str):
        """
        Exporte les données nettoyées vers un fichier CSV
        
        Args:
            filepath: Chemin de destination
        """
        if self.cleaned_data is None:
            raise ValueError("Aucune donnee nettoyee disponible")
        
        self.cleaned_data.to_csv(filepath, index=False, encoding='utf-8')
        print(f"Donnees nettoyees exportees vers: {filepath}")