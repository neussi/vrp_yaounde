"""
Point d'entrée principal pour l'algorithme VRP Yaoundé

Ce script orchestre l'ensemble du processus:
1. Chargement et preprocessing des données
2. Configuration des paramètres
3. Exécution de l'algorithme ALNS
4. Génération des résultats et visualisations
5. Export des solutions

Utilisation:
    python3 main.py --data data/trajets_yde.csv --iterations 5000 --output results/
"""

import os
import sys
import argparse
import json
import time
from datetime import datetime
import numpy as np
import pandas as pd

# Import des modules du projet
from src.preprocessing import DataPreprocessor
from src.cost_matrix import CostMatrixBuilder
from src.models.solution import Solution
from src.models.vrp_model import VRPModel, VRPParameters
from src.algorithms.alns import ALNS
from src.algorithms.initial_solution import InitialSolutionGenerator
from src.utils.cost_functions import CostCalculator
from src.utils.feasibility import FeasibilityChecker
from src.utils.visualization import VRPVisualizer
from src.utils.config import Config


def parse_arguments():
    """
    Parse les arguments de la ligne de commande
    
    Returns:
        Arguments parsés
    """
    parser = argparse.ArgumentParser(
        description='Résolution du Vehicle Routing Problem pour Yaoundé',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Arguments obligatoires
    parser.add_argument(
        '--data',
        type=str,
        default='data/trajets_yde.csv',
        help='Chemin vers le fichier CSV des données'
    )
    
    # Paramètres de l\'algorithme
    parser.add_argument(
        '--iterations',
        type=int,
        default=5000,
        help='Nombre maximum d\'itérations ALNS'
    )
    
    parser.add_argument(
        '--max-vehicles',
        type=int,
        default=20,
        help='Nombre maximum de véhicules disponibles'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Graine aléatoire pour reproductibilité'
    )
    
    # Paramètres de température (Simulated Annealing)
    parser.add_argument(
        '--initial-temperature',
        type=float,
        default=1000.0,
        help='Température initiale pour SA'
    )
    
    parser.add_argument(
        '--cooling-rate',
        type=float,
        default=0.995,
        help='Taux de refroidissement (alpha)'
    )
    
    # Sorties
    parser.add_argument(
        '--output',
        type=str,
        default='results/',
        help='Répertoire de sortie pour les résultats'
    )
    
    parser.add_argument(
        '--no-viz',
        action='store_true',
        help='Désactiver la génération de visualisations'
    )
    
    parser.add_argument(
        '--export-model',
        action='store_true',
        help='Exporter le modèle mathématique (.lp)'
    )
    
    return parser.parse_args()


def setup_directories(output_dir):
    """
    Crée les répertoires nécessaires pour les sorties
    
    Args:
        output_dir: Répertoire de base pour les sorties
    """
    directories = [
        output_dir,
        os.path.join(output_dir, 'graphiques'),
        os.path.join(output_dir, 'logs'),
        os.path.join(output_dir, 'solutions')
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print(f"Repertoires de sortie crees dans: {output_dir}")


def load_and_preprocess_data(data_path):
    """
    Charge et prétraite les données
    
    Args:
        data_path: Chemin vers le fichier CSV
        
    Returns:
        Tuple (df_clean, preprocessor)
    """
    print("\n" + "="*70)
    print("PHASE 1: CHARGEMENT ET PREPROCESSING DES DONNEES")
    print("="*70)
    
    # Vérifier que le fichier existe
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Fichier de donnees introuvable: {data_path}")
    
    # Charger les données
    print(f"\nChargement des donnees depuis: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Donnees chargees: {len(df)} trajets")
    
    # Prétraiter
    preprocessor = DataPreprocessor(df)
    df_clean = preprocessor.clean_data()
    
    print(f"\nPreprocessing termine:")
    print(f"  - Donnees nettoyees: {len(df_clean)} lignes")
    print(f"  - Clients uniques: {df_clean['Station Arrivée'].nunique()}")
    print(f"  - Distance totale reseau: {df_clean['Distance (km)'].sum():.1f} km")
    
    return df_clean, preprocessor


def build_cost_matrix(df_clean, config, preprocessor):
    """
    Construit la matrice de coûts

    Args:
        df_clean: DataFrame nettoyé
        config: Configuration
        preprocessor: Instance du DataPreprocessor

    Returns:
        cost_matrix (numpy array)
    """
    print("\n" + "="*70)
    print("PHASE 2: CONSTRUCTION DE LA MATRICE DE COUTS")
    print("="*70)

    # Préparer les données au format attendu par CostMatrixBuilder
    processed_data = {
        'data': df_clean,
        'station_map': preprocessor.station_mapping,
        'n_stations': len(preprocessor.station_mapping) - 1  # Exclure le dépôt
    }

    builder = CostMatrixBuilder(processed_data)
    matrices = builder.build_all_matrices()

    # Utiliser la matrice de coûts de la période 'matin' par défaut
    cost_matrix = matrices['cost']['matin']

    print(f"\nMatrice de couts construite: {cost_matrix.shape}")
    print(f"  - Cout minimum (non-zero): {np.min(cost_matrix[cost_matrix > 0]):.2f}")
    print(f"  - Cout maximum: {np.max(cost_matrix):.2f}")
    print(f"  - Cout moyen: {np.mean(cost_matrix[cost_matrix > 0]):.2f}")

    return cost_matrix, builder


def setup_vrp_parameters(df_clean, cost_matrix, config):
    """
    Configure les paramètres du modèle VRP
    
    Args:
        df_clean: DataFrame nettoyé
        cost_matrix: Matrice des coûts
        config: Configuration
        
    Returns:
        VRPParameters
    """
    print("\n" + "="*70)
    print("PHASE 3: CONFIGURATION DU MODELE VRP")
    print("="*70)
    
    # Nombre de clients (exclure dépôt)
    num_customers = cost_matrix.shape[0] - 1
    
    # Demandes des clients (simulation si pas dans les données)
    demands = {}
    for i in range(1, num_customers + 1):
        # Simuler une demande aléatoire entre 50 et 200 unités
        demands[i] = np.random.randint(50, 200)
    
    # Fenêtres temporelles (extraire depuis les données ou simuler)
    time_windows = {}
    for i in range(1, num_customers + 1):
        # Fenêtre par défaut: 8h-18h
        time_windows[i] = (8.0, 18.0)
    
    # Temps de service (30 minutes par défaut)
    service_times = {i: 0.5 for i in range(num_customers + 1)}
    
    # Capacités des véhicules
    vehicle_capacities = {
        'Camionnette': 1000,
        'Camion moyen': 2000,
        '4x4': 1500
    }
    
    vehicle_types = list(vehicle_capacities.keys())
    
    # Coûts fixes des véhicules
    fixed_vehicle_costs = {
        'Camionnette': 5000,
        'Camion moyen': 8000,
        '4x4': 10000
    }
    
    # Créer les paramètres
    parameters = VRPParameters(
        num_customers=num_customers,
        num_vehicles=config.max_vehicles,
        distance_matrix=cost_matrix,
        cost_matrix=cost_matrix,
        informal_fees=np.zeros_like(cost_matrix),
        toll_fees=np.zeros_like(cost_matrix),
        demands=demands,
        time_windows=time_windows,
        service_times=service_times,
        vehicle_capacities=vehicle_capacities,
        vehicle_types=vehicle_types,
        fixed_vehicle_costs=fixed_vehicle_costs,
        road_conditions={},
        rain_practicability={},
        hourly_restrictions={},
        vehicle_requirements={},
        parking_difficulty={},
        security_level={},
        fuel_cost_per_km=config.fuel_cost_per_km,
        average_speed=config.average_speed
    )
    
    print(f"\nParametres du modele:")
    print(f"  - Clients: {parameters.num_customers}")
    print(f"  - Vehicules disponibles: {parameters.num_vehicles}")
    print(f"  - Types de vehicules: {len(parameters.vehicle_types)}")
    print(f"  - Demande totale: {sum(parameters.demands.values())} unites")
    print(f"  - Capacite totale: {sum(parameters.vehicle_capacities.values())} unites")
    
    return parameters


def solve_vrp(parameters, config):
    """
    Résout le VRP avec l'algorithme ALNS
    
    Args:
        parameters: Paramètres du VRP
        config: Configuration
        
    Returns:
        Tuple (best_solution, statistics)
    """
    print("\n" + "="*70)
    print("PHASE 4: RESOLUTION DU VRP AVEC ALNS")
    print("="*70)
    
    # Initialiser l'algorithme ALNS
    alns = ALNS(
        cost_matrix=parameters.cost_matrix,
        demands=parameters.demands,
        time_windows=parameters.time_windows,
        vehicle_capacities=parameters.vehicle_capacities,
        config=config.__dict__
    )
    
    # Résoudre
    best_solution, stats = alns.solve(max_iterations=config.max_iterations)
    
    return best_solution, stats


def validate_and_analyze_solution(solution, parameters, config):
    """
    Valide et analyse la solution obtenue
    
    Args:
        solution: Solution à valider
        parameters: Paramètres du VRP
        config: Configuration
        
    Returns:
        Dictionnaire de résultats d'analyse
    """
    print("\n" + "="*70)
    print("PHASE 5: VALIDATION ET ANALYSE DE LA SOLUTION")
    print("="*70)
    
    # Créer le vérificateur de faisabilité
    feasibility_checker = FeasibilityChecker(
        cost_matrix=parameters.cost_matrix,
        demands=parameters.demands,
        time_windows=parameters.time_windows,
        vehicle_capacities=parameters.vehicle_capacities,
        route_data={},
        config=config.__dict__
    )
    
    # Vérifier la faisabilité
    is_feasible, feasibility_report = feasibility_checker.check_solution(
        solution, detailed=True
    )
    
    print(f"\nFaisabilite: {'OUI' if is_feasible else 'NON'}")
    if not is_feasible:
        print(f"Violations detectees: {feasibility_report['total_violations']}")
        if feasibility_report['capacity_violations']:
            print(f"  - Capacite: {len(feasibility_report['capacity_violations'])}")
        if feasibility_report['time_violations']:
            print(f"  - Temps: {len(feasibility_report['time_violations'])}")
        if feasibility_report['restriction_violations']:
            print(f"  - Restrictions: {len(feasibility_report['restriction_violations'])}")
    
    # Calculer les métriques de qualité
    quality_metrics = feasibility_checker.get_solution_quality_metrics(solution)
    
    print(f"\nMetriques de qualite:")
    print(f"  - Score de faisabilite: {quality_metrics['feasibility_score']:.1f}/100")
    print(f"  - Utilisation capacite: {quality_metrics['capacity_utilization']:.1f}%")
    print(f"  - Equilibre des routes: {quality_metrics['route_balance']:.1f}/100")
    print(f"  - Conformite fenetres temp: {quality_metrics['time_window_compliance']:.1f}%")
    
    # Suggestions d'amélioration
    suggestions = feasibility_checker.suggest_improvements(solution)
    if suggestions:
        print(f"\nSuggestions d'amelioration:")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"  {i}. {suggestion}")
    
    return {
        'is_feasible': is_feasible,
        'feasibility_report': feasibility_report,
        'quality_metrics': quality_metrics,
        'suggestions': suggestions
    }


def generate_visualizations(solution, stats, analysis_results, output_dir, config):
    """
    Génère toutes les visualisations
    
    Args:
        solution: Solution à visualiser
        stats: Statistiques d'exécution
        analysis_results: Résultats de l'analyse
        output_dir: Répertoire de sortie
        config: Configuration
    """
    print("\n" + "="*70)
    print("PHASE 6: GENERATION DES VISUALISATIONS")
    print("="*70)
    
    viz_dir = os.path.join(output_dir, 'graphiques')
    visualizer = VRPVisualizer()
    
    # 1. Carte des routes
    print("\n1. Generation de la carte des routes...")
    visualizer.plot_routes(
        solution,
        save_path=os.path.join(viz_dir, 'routes_map.png'),
        title='Routes Optimales - VRP Yaounde'
    )
    
    # 2. Convergence de l'algorithme
    print("2. Generation du graphique de convergence...")
    visualizer.plot_convergence(
        stats['history'],
        save_path=os.path.join(viz_dir, 'convergence.png')
    )
    
    # 3. Utilisation des véhicules
    print("3. Generation du graphique d'utilisation des vehicules...")
    visualizer.plot_vehicle_utilization(
        solution,
        save_path=os.path.join(viz_dir, 'vehicle_utilization.png')
    )
    
    # 4. Violations de contraintes
    print("4. Generation du graphique des contraintes...")
    visualizer.plot_constraint_violations(
        analysis_results['feasibility_report'],
        save_path=os.path.join(viz_dir, 'constraint_violations.png')
    )
    
    # 5. Rapport complet
    print("5. Generation du rapport complet...")
    visualizer.create_summary_report(
        solution,
        stats,
        analysis_results['feasibility_report'],
        save_path=os.path.join(viz_dir, 'summary_report.png')
    )
    
    print(f"\nToutes les visualisations sauvegardees dans: {viz_dir}")


def export_results(solution, stats, analysis_results, parameters, output_dir):
    """
    Exporte les résultats dans différents formats
    
    Args:
        solution: Solution à exporter
        stats: Statistiques
        analysis_results: Résultats d'analyse
        parameters: Paramètres du VRP
        output_dir: Répertoire de sortie
    """
    print("\n" + "="*70)
    print("PHASE 7: EXPORT DES RESULTATS")
    print("="*70)
    
    solutions_dir = os.path.join(output_dir, 'solutions')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Export JSON de la solution
    print("\n1. Export JSON de la solution...")
    solution_dict = {
        'metadata': {
            'timestamp': timestamp,
            'num_vehicles': solution.num_vehicles_used,
            'total_cost': solution.total_cost,
            'total_distance': solution.total_distance,
            'is_feasible': analysis_results['is_feasible']
        },
        'routes': []
    }
    
    for route in solution.routes:
        if route.is_empty():
            continue
        
        route_dict = {
            'vehicle_id': route.vehicle_id,
            'vehicle_type': route.vehicle_type,
            'customers': route.customers,
            'load': route.load,
            'capacity': route.capacity,
            'distance': route.total_distance,
            'cost': route.total_cost
        }
        solution_dict['routes'].append(route_dict)
    
    json_path = os.path.join(solutions_dir, f'solution_{timestamp}.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(solution_dict, f, indent=2, ensure_ascii=False)
    print(f"   Solution exportee: {json_path}")
    
    # 2. Export CSV des routes
    print("2. Export CSV des routes...")
    routes_data = []
    for route in solution.routes:
        if route.is_empty():
            continue
        
        for i, customer in enumerate(route.get_customers_only()):
            routes_data.append({
                'Route_ID': route.vehicle_id,
                'Vehicle_Type': route.vehicle_type,
                'Sequence': i + 1,
                'Customer_ID': customer,
                'Route_Load': route.load,
                'Route_Capacity': route.capacity,
                'Route_Distance': route.total_distance,
                'Route_Cost': route.total_cost
            })
    
    df_routes = pd.DataFrame(routes_data)
    csv_path = os.path.join(solutions_dir, f'routes_{timestamp}.csv')
    df_routes.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"   Routes exportees: {csv_path}")
    
    # 3. Export log des statistiques
    print("3. Export des statistiques...")
    logs_dir = os.path.join(output_dir, 'logs')
    log_path = os.path.join(logs_dir, f'execution_log_{timestamp}.txt')
    
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("RAPPORT D'EXECUTION - VRP YAOUNDE\n")
        f.write("="*70 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("PARAMETRES:\n")
        f.write(f"  Nombre de clients: {parameters.num_customers}\n")
        f.write(f"  Vehicules disponibles: {parameters.num_vehicles}\n")
        f.write(f"  Iterations: {stats['iterations']}\n\n")
        
        f.write("SOLUTION:\n")
        f.write(f"  Cout total: {solution.total_cost:,.0f} FCFA\n")
        f.write(f"  Distance totale: {solution.total_distance:.1f} km\n")
        f.write(f"  Vehicules utilises: {solution.num_vehicles_used}\n")
        f.write(f"  Clients desservis: {len(solution.get_all_customers())}\n")
        f.write(f"  Temps d'execution: {stats['execution_time']:.2f} secondes\n")
        f.write(f"  Amelioration: {stats['improvement_percent']:.2f}%\n\n")
        
        f.write("FAISABILITE:\n")
        f.write(f"  Statut: {'FAISABLE' if analysis_results['is_feasible'] else 'INFAISABLE'}\n")
        f.write(f"  Score: {analysis_results['quality_metrics']['feasibility_score']:.1f}/100\n\n")
        
        f.write("DETAILS DES ROUTES:\n")
        for route in solution.routes:
            if route.is_empty():
                continue
            f.write(f"\n  Route {route.vehicle_id} ({route.vehicle_type}):\n")
            f.write(f"    Clients: {route.get_customers_only()}\n")
            f.write(f"    Charge: {route.load:.1f}/{route.capacity} unites\n")
            f.write(f"    Distance: {route.total_distance:.1f} km\n")
            f.write(f"    Cout: {route.total_cost:,.0f} FCFA\n")
    
    print(f"   Log d'execution: {log_path}")
    
    print(f"\nTous les resultats exportes dans: {output_dir}")


def main():
    """
    Fonction principale
    """
    # Banner de démarrage
    print("\n" + "="*70)
    print("  VEHICLE ROUTING PROBLEM - YAOUNDE")
    print("  Optimisation des tournees de livraison")
    print("  Algorithme: Adaptive Large Neighborhood Search (ALNS)")
    print("="*70)
    
    # Parser les arguments
    args = parse_arguments()
    
    # Créer la configuration
    config = Config(
        max_vehicles=args.max_vehicles,
        max_iterations=args.iterations,
        seed=args.seed,
        initial_temperature=args.initial_temperature,
        cooling_rate=args.cooling_rate
    )
    
    # Configurer numpy seed
    np.random.seed(config.seed)
    
    print(f"\nConfiguration:")
    print(f"  - Fichier de donnees: {args.data}")
    print(f"  - Iterations max: {args.iterations}")
    print(f"  - Vehicules max: {args.max_vehicles}")
    print(f"  - Seed: {args.seed}")
    print(f"  - Repertoire de sortie: {args.output}")
    
    # Créer les répertoires de sortie
    setup_directories(args.output)
    
    try:
        # PHASE 1: Charger et prétraiter les données
        df_clean, preprocessor = load_and_preprocess_data(args.data)
        
        # PHASE 2: Construire la matrice de coûts
        cost_matrix, builder = build_cost_matrix(df_clean, config, preprocessor)
        
        # PHASE 3: Configurer les paramètres du VRP
        parameters = setup_vrp_parameters(df_clean, cost_matrix, config)
        
        # Export du modèle mathématique si demandé
        if args.export_model:
            print("\nExport du modele mathematique...")
            vrp_model = VRPModel(parameters)
            vrp_model.print_model_summary()
            model_path = os.path.join(args.output, 'vrp_model.lp')
            vrp_model.export_to_lp(model_path)
        
        # PHASE 4: Résoudre le VRP
        start_time = time.time()
        best_solution, stats = solve_vrp(parameters, config)
        stats['execution_time'] = time.time() - start_time
        
        # PHASE 5: Valider et analyser
        analysis_results = validate_and_analyze_solution(
            best_solution, parameters, config
        )
        
        # PHASE 6: Générer les visualisations
        if not args.no_viz:
            generate_visualizations(
                best_solution, stats, analysis_results, 
                args.output, config
            )
        
        # PHASE 7: Exporter les résultats
        export_results(
            best_solution, stats, analysis_results, 
            parameters, args.output
        )
        
        # Message de succès
        print("\n" + "="*70)
        print("EXECUTION TERMINEE AVEC SUCCES!")
        print("="*70)
        print(f"\nCout final: {best_solution.total_cost:,.0f} FCFA")
        print(f"Distance totale: {best_solution.total_distance:.1f} km")
        print(f"Vehicules utilises: {best_solution.num_vehicles_used}")
        print(f"Temps d'execution: {stats['execution_time']:.2f} secondes")
        print(f"\nResultats disponibles dans: {args.output}")
        
    except Exception as e:
        print(f"\n[ERREUR] Une erreur s'est produite: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()