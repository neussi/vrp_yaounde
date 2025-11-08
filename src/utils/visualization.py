"""
Visualisation des solutions du VRP

Génère des graphiques et cartes pour analyser les résultats:
- Cartes des routes
- Graphiques d'évolution
- Diagrammes de Gantt
- Statistiques visuelles
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
from typing import Dict, List, Tuple
import seaborn as sns

# Configuration du style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

class VRPVisualizer:
    """
    Classe pour visualiser les solutions du VRP
    
    Fonctionnalités:
    - Carte des routes
    - Evolution du coût
    - Utilisation des véhicules
    - Respect des contraintes
    - Diagrammes temporels
    """
    
    def __init__(self, coordinates: Dict[int, Tuple[float, float]] = None):
        """
        Initialisation du visualiseur
        
        Args:
            coordinates: Dictionnaire {node_id: (x, y)} des coordonnées
                        Si None, génère des coordonnées aléatoires
        """
        self.coordinates = coordinates
        self.colors = plt.cm.tab20.colors  # Palette de 20 couleurs
    
    def plot_routes(self, solution, save_path: str = None, title: str = "Routes Optimales"):
        """
        Dessine les routes sur une carte
        
        Chaque route a une couleur différente.
        Le dépôt est marqué d'une étoile.
        
        Args:
            solution: Solution à visualiser
            save_path: Chemin pour sauvegarder l'image
            title: Titre du graphique
        """
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Si pas de coordonnées, en générer
        if self.coordinates is None:
            self._generate_random_coordinates(solution)
        
        # Dessiner chaque route
        for idx, route in enumerate(solution.routes):
            if route.is_empty():
                continue
            
            color = self.colors[idx % len(self.colors)]
            customers = route.customers
            
            # Tracer les segments
            for i in range(len(customers) - 1):
                from_node = customers[i]
                to_node = customers[i + 1]
                
                x_coords = [self.coordinates[from_node][0], 
                           self.coordinates[to_node][0]]
                y_coords = [self.coordinates[from_node][1], 
                           self.coordinates[to_node][1]]
                
                # Tracer l'arc
                ax.plot(x_coords, y_coords, 'o-', color=color, 
                       linewidth=2, markersize=8, alpha=0.7,
                       label=f'Route {idx+1}' if i == 0 else '')
                
                # Ajouter une flèche pour indiquer la direction
                if i > 0 and i < len(customers) - 1:
                    dx = x_coords[1] - x_coords[0]
                    dy = y_coords[1] - y_coords[0]
                    ax.arrow(x_coords[0] + 0.3*dx, y_coords[0] + 0.3*dy,
                            0.2*dx, 0.2*dy, head_width=0.3, 
                            head_length=0.2, fc=color, ec=color, alpha=0.5)
        
        # Marquer le dépôt
        depot_coords = self.coordinates[0]
        ax.plot(depot_coords[0], depot_coords[1], 'r*', 
               markersize=25, label='Depot Central', zorder=5)
        
        # Annoter les clients
        for node, (x, y) in self.coordinates.items():
            if node != 0:
                ax.annotate(str(node), (x, y), fontsize=8, 
                           ha='center', va='center',
                           bbox=dict(boxstyle='circle', fc='white', 
                                   ec='black', alpha=0.7))
        
        ax.set_xlabel('Longitude (coordonnees relatives)', fontsize=12)
        ax.set_ylabel('Latitude (coordonnees relatives)', fontsize=12)
        ax.set_title(f'{title}\n{solution.num_vehicles_used} vehicules, '
                    f'{solution.total_distance:.1f} km, '
                    f'{solution.total_cost:,.0f} FCFA', 
                    fontsize=14, fontweight='bold')
        
        # Légende (limiter à 10 routes max pour lisibilité)
        handles, labels = ax.get_legend_handles_labels()
        if len(handles) > 11:  # 10 routes + dépôt
            handles = handles[:11]
            labels = labels[:11]
        ax.legend(handles, labels, loc='upper right', fontsize=9)
        
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Carte des routes sauvegardee: {save_path}")
        
        plt.show()
    
    def plot_convergence(self, history: Dict, save_path: str = None):
        """
        Trace l'évolution du coût au cours des itérations
        
        Args:
            history: Dictionnaire avec 'best_costs' et 'current_costs'
            save_path: Chemin pour sauvegarder l'image
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        iterations = range(len(history['best_costs']))
        
        # Graphique 1: Evolution des coûts
        ax1.plot(iterations, history['best_costs'], 'b-', 
                linewidth=2, label='Meilleur cout global')
        ax1.plot(iterations, history['current_costs'], 'r-', 
                alpha=0.5, linewidth=1, label='Cout courant')
        
        ax1.set_xlabel('Iteration', fontsize=12)
        ax1.set_ylabel('Cout (FCFA)', fontsize=12)
        ax1.set_title('Convergence de l\'Algorithme ALNS', 
                     fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Calculer l'amélioration
        initial_cost = history['best_costs'][0]
        final_cost = history['best_costs'][-1]
        improvement = ((initial_cost - final_cost) / initial_cost) * 100
        
        ax1.text(0.02, 0.98, 
                f'Amelioration: {improvement:.2f}%\n'
                f'Cout initial: {initial_cost:,.0f} FCFA\n'
                f'Cout final: {final_cost:,.0f} FCFA',
                transform=ax1.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Graphique 2: Evolution de la température
        if 'temperatures' in history:
            ax2.plot(iterations, history['temperatures'], 'g-', linewidth=2)
            ax2.set_xlabel('Iteration', fontsize=12)
            ax2.set_ylabel('Temperature', fontsize=12)
            ax2.set_title('Evolution de la Temperature (Simulated Annealing)', 
                         fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.set_yscale('log')  # Échelle logarithmique pour mieux voir
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Graphique de convergence sauvegarde: {save_path}")
        
        plt.show()
    
    def plot_vehicle_utilization(self, solution, save_path: str = None):
        """
        Visualise l'utilisation des véhicules (charge et distance)
        
        Args:
            solution: Solution à analyser
            save_path: Chemin pour sauvegarder l'image
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Préparer les données
        vehicle_ids = []
        loads = []
        capacities = []
        distances = []
        
        for route in solution.routes:
            if route.is_empty():
                continue
            
            vehicle_ids.append(f"V{route.vehicle_id}\n{route.vehicle_type}")
            loads.append(route.load)
            capacities.append(route.capacity)
            distances.append(route.total_distance)
        
        x = np.arange(len(vehicle_ids))
        
        # Graphique 1: Utilisation de la capacité
        bars1 = ax1.bar(x, loads, width=0.6, label='Charge actuelle', color='steelblue')
        ax1.plot(x, capacities, 'r--', linewidth=2, marker='o', 
                label='Capacite maximale', markersize=8)
        
        # Calculer et afficher le taux d'utilisation
        for i, (load, capacity) in enumerate(zip(loads, capacities)):
            utilization = (load / capacity * 100) if capacity > 0 else 0
            ax1.text(i, load + max(capacities)*0.02, f'{utilization:.0f}%', 
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax1.set_xlabel('Vehicule', fontsize=12)
        ax1.set_ylabel('Charge (unites)', fontsize=12)
        ax1.set_title('Utilisation de la Capacite des Vehicules', 
                     fontsize=13, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(vehicle_ids, fontsize=9)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Graphique 2: Distance parcourue par véhicule
        bars2 = ax2.bar(x, distances, width=0.6, color='coral')
        
        # Afficher les valeurs
        for i, distance in enumerate(distances):
            ax2.text(i, distance + max(distances)*0.02, f'{distance:.1f}', 
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax2.set_xlabel('Vehicule', fontsize=12)
        ax2.set_ylabel('Distance (km)', fontsize=12)
        ax2.set_title('Distance Parcourue par Vehicule', 
                     fontsize=13, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(vehicle_ids, fontsize=9)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Statistiques globales
        avg_utilization = np.mean([l/c*100 for l, c in zip(loads, capacities)])
        avg_distance = np.mean(distances)
        
        fig.text(0.5, 0.02, 
                f'Statistiques: Utilisation moyenne = {avg_utilization:.1f}% | '
                f'Distance moyenne = {avg_distance:.1f} km | '
                f'Total vehicules = {len(vehicle_ids)}',
                ha='center', fontsize=11, 
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout(rect=[0, 0.04, 1, 1])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Graphique d'utilisation sauvegarde: {save_path}")
        
        plt.show()
    
    def plot_gantt_chart(self, solution, time_windows, cost_matrix, 
                        save_path: str = None):
        """
        Crée un diagramme de Gantt des routes dans le temps
        
        Montre quand chaque véhicule visite chaque client
        
        Args:
            solution: Solution à visualiser
            time_windows: Fenêtres temporelles des clients
            cost_matrix: Matrice pour calcul des temps
            save_path: Chemin pour sauvegarder l'image
        """
        from src.constraints.time_windows import TimeWindowManager
        
        fig, ax = plt.subplots(figsize=(16, max(8, len(solution.routes))))
        
        tw_manager = TimeWindowManager(time_windows, {'service_time': 0.5, 'average_speed': 25})
        
        y_pos = 0
        yticks = []
        yticklabels = []
        
        for route in solution.routes:
            if route.is_empty():
                continue
            
            # Calculer les temps d'arrivée
            schedule = tw_manager.calculate_arrival_times(route.customers, cost_matrix)
            
            # Tracer les segments de temps
            for customer, info in schedule.items():
                if customer == 0:  # Ignorer le dépôt
                    continue
                
                start = info['arrival_time']
                service_start = info['start_service']
                end = info['departure_time']
                
                # Temps d'attente (en rouge clair)
                if info['waiting_time'] > 0:
                    ax.barh(y_pos, info['waiting_time'], 
                           left=start, height=0.5, 
                           color='lightcoral', alpha=0.7, 
                           edgecolor='black', linewidth=0.5)
                
                # Temps de service (en vert)
                ax.barh(y_pos, end - service_start, 
                       left=service_start, height=0.5,
                       color='lightgreen', alpha=0.8,
                       edgecolor='black', linewidth=0.5)
                
                # Annoter le client
                ax.text(service_start + (end - service_start)/2, y_pos,
                       f'C{customer}', ha='center', va='center', 
                       fontsize=8, fontweight='bold')
                
                # Marquer la fenêtre temporelle
                if customer in time_windows:
                    tw_start, tw_end = time_windows[customer]
                    ax.plot([tw_start, tw_end], [y_pos, y_pos], 
                           'b-', linewidth=3, alpha=0.3)
            
            yticks.append(y_pos)
            yticklabels.append(f'Route {route.vehicle_id}\n{route.vehicle_type}')
            y_pos += 1
        
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels, fontsize=9)
        ax.set_xlabel('Temps (heures)', fontsize=12)
        ax.set_title('Diagramme de Gantt - Planification Temporelle des Routes', 
                    fontsize=14, fontweight='bold')
        ax.set_xlim(5, 22)  # 5h à 22h
        ax.grid(True, alpha=0.3, axis='x')
        
        # Légende
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='lightcoral', alpha=0.7, label='Temps d\'attente'),
            Patch(facecolor='lightgreen', alpha=0.8, label='Temps de service'),
            plt.Line2D([0], [0], color='b', linewidth=3, alpha=0.3, label='Fenetre temporelle')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Diagramme de Gantt sauvegarde: {save_path}")
        
        plt.show()
    
    def plot_cost_breakdown(self, solution, cost_calculator, save_path: str = None):
        """
        Décompose les coûts par catégorie (pie chart + bar chart)
        
        Args:
            solution: Solution à analyser
            cost_calculator: Instance de CostCalculator
            save_path: Chemin pour sauvegarder l'image
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Calculer les coûts détaillés
        total_travel = 0
        total_fixed = 0
        total_informal = 0
        total_tolls = 0
        
        for route in solution.routes:
            if route.is_empty():
                continue
            
            costs = cost_calculator.calculate_route_cost(
                route.customers, route.vehicle_type, 
                include_penalties=False
            )
            
            total_travel += costs['travel_cost']
            total_fixed += costs['fixed_cost']
            total_informal += costs['informal_fees']
            total_tolls += costs['tolls']
        
        # Graphique 1: Pie chart
        categories = ['Carburant\n& Transport', 'Couts Fixes\nVehicules', 
                     'Frais\nInformels', 'Peages']
        values = [total_travel, total_fixed, total_informal, total_tolls]
        colors_pie = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
        
        # Filtrer les valeurs nulles
        filtered_data = [(cat, val, col) for cat, val, col in zip(categories, values, colors_pie) if val > 0]
        if filtered_data:
            categories, values, colors_pie = zip(*filtered_data)
        
        wedges, texts, autotexts = ax1.pie(values, labels=categories, colors=colors_pie,
                                           autopct='%1.1f%%', startangle=90,
                                           textprops={'fontsize': 11, 'fontweight': 'bold'})
        
        ax1.set_title('Repartition des Couts', fontsize=13, fontweight='bold')
        
        # Graphique 2: Bar chart
        x_pos = np.arange(len(categories))
        bars = ax2.bar(x_pos, values, color=colors_pie, edgecolor='black', linewidth=1.5)
        
        # Afficher les valeurs
        for i, (bar, value) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.02,
                    f'{value:,.0f}\nFCFA',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(categories, fontsize=10)
        ax2.set_ylabel('Cout (FCFA)', fontsize=12)
        ax2.set_title('Detail des Couts par Categorie', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Total
        total_cost = sum(values)
        fig.text(0.5, 0.02, 
                f'COUT TOTAL: {total_cost:,.0f} FCFA',
                ha='center', fontsize=13, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Decomposition des couts sauvegardee: {save_path}")
        
        plt.show()
    
    def plot_constraint_violations(self, feasibility_report: Dict, 
                                   save_path: str = None):
        """
        Visualise les violations de contraintes
        
        Args:
            feasibility_report: Rapport de faisabilité
            save_path: Chemin pour sauvegarder l'image
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        categories = ['Capacite', 'Fenetres\nTemporelles', 'Restrictions\nHoraires']
        violations = [
            len(feasibility_report.get('capacity_violations', [])),
            len(feasibility_report.get('time_violations', [])),
            len(feasibility_report.get('restriction_violations', []))
        ]
        
        colors_viol = ['red' if v > 0 else 'green' for v in violations]
        
        bars = ax.bar(categories, violations, color=colors_viol, 
                     edgecolor='black', linewidth=2)
        
        # Afficher les nombres
        for bar, value in zip(bars, violations):
            height = bar.get_height()
            label = 'OK' if value == 0 else str(value)
            color_text = 'green' if value == 0 else 'red'
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   label, ha='center', va='bottom', 
                   fontsize=14, fontweight='bold', color=color_text)
        
        ax.set_ylabel('Nombre de Violations', fontsize=12)
        ax.set_title('Respect des Contraintes', fontsize=14, fontweight='bold')
        ax.set_ylim(0, max(violations) + 2 if max(violations) > 0 else 2)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Statut global
        is_feasible = feasibility_report.get('feasible', False)
        status_text = 'SOLUTION FAISABLE' if is_feasible else 'SOLUTION INFAISABLE'
        status_color = 'green' if is_feasible else 'red'
        
        ax.text(0.5, 0.95, status_text, transform=ax.transAxes,
               fontsize=16, fontweight='bold', ha='center', va='top',
               bbox=dict(boxstyle='round', facecolor=status_color, 
                        alpha=0.7, edgecolor='black', linewidth=2))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Graphique des violations sauvegarde: {save_path}")
        
        plt.show()
    
    def create_summary_report(self, solution, stats: Dict, 
                             feasibility_report: Dict,
                             save_path: str = None):
        """
        Crée un rapport visuel complet sur une page
        
        Args:
            solution: Solution à résumer
            stats: Statistiques d'exécution
            feasibility_report: Rapport de faisabilité
            save_path: Chemin pour sauvegarder l'image
        """
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Titre principal
        fig.suptitle('RAPPORT DE SOLUTION - VRP YAOUNDE', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        # 1. Informations générales (texte)
        ax_info = fig.add_subplot(gs[0, :])
        ax_info.axis('off')
        
        info_text = f"""
        STATISTIQUES GENERALES
        
        Cout Total: {solution.total_cost:,.0f} FCFA
        Distance Totale: {solution.total_distance:.1f} km
        Nombre de Vehicules Utilises: {solution.num_vehicles_used}
        Clients Desservis: {len(solution.get_all_customers())}
        Clients Non Desservis: {len(solution.unassigned_customers)}
        
        Temps d'Execution: {stats.get('execution_time', 0):.2f} secondes
        Iterations: {stats.get('iterations', 0)}
        Amelioration: {stats.get('improvement_percent', 0):.2f}%
        """
        
        ax_info.text(0.1, 0.5, info_text, fontsize=12, 
                    family='monospace', va='center',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        # Statut de faisabilité
        is_feasible = feasibility_report.get('feasible', False)
        status = 'FAISABLE' if is_feasible else 'INFAISABLE'
        status_color = 'green' if is_feasible else 'red'
        
        ax_info.text(0.7, 0.5, f'STATUT:\n{status}', 
                    fontsize=16, fontweight='bold', ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor=status_color, 
                            alpha=0.7, edgecolor='black', linewidth=3))
        
        # 2. Convergence (ligne 2, colonne 1)
        ax_conv = fig.add_subplot(gs[1, 0])
        history = stats.get('history', {})
        if 'best_costs' in history:
            iterations = range(len(history['best_costs']))
            ax_conv.plot(iterations, history['best_costs'], 'b-', linewidth=2)
            ax_conv.set_xlabel('Iteration', fontsize=9)
            ax_conv.set_ylabel('Cout (FCFA)', fontsize=9)
            ax_conv.set_title('Convergence', fontsize=11, fontweight='bold')
            ax_conv.grid(True, alpha=0.3)
        
        # 3. Utilisation véhicules (ligne 2, colonne 2)
        ax_util = fig.add_subplot(gs[1, 1])
        vehicle_types = {}
        for route in solution.routes:
            if not route.is_empty():
                vtype = route.vehicle_type
                vehicle_types[vtype] = vehicle_types.get(vtype, 0) + 1
        
        if vehicle_types:
            ax_util.pie(vehicle_types.values(), labels=vehicle_types.keys(),
                       autopct='%1.0f%%', startangle=90)
            ax_util.set_title('Types de Vehicules', fontsize=11, fontweight='bold')
        
        # 4. Violations (ligne 2, colonne 3)
        ax_viol = fig.add_subplot(gs[1, 2])
        violations = [
            len(feasibility_report.get('capacity_violations', [])),
            len(feasibility_report.get('time_violations', [])),
            len(feasibility_report.get('restriction_violations', []))
        ]
        categories = ['Capacite', 'Temps', 'Restrictions']
        colors = ['red' if v > 0 else 'green' for v in violations]
        
        ax_viol.bar(categories, violations, color=colors, edgecolor='black')
        ax_viol.set_ylabel('Violations', fontsize=9)
        ax_viol.set_title('Contraintes', fontsize=11, fontweight='bold')
        ax_viol.grid(True, alpha=0.3, axis='y')
        
        # 5. Distribution distances (ligne 3, colonne 1)
        ax_dist = fig.add_subplot(gs[2, 0])
        distances = [r.total_distance for r in solution.routes if not r.is_empty()]
        if distances:
            ax_dist.hist(distances, bins=min(10, len(distances)), 
                        color='skyblue', edgecolor='black')
            ax_dist.set_xlabel('Distance (km)', fontsize=9)
            ax_dist.set_ylabel('Frequence', fontsize=9)
            ax_dist.set_title('Distribution Distances', fontsize=11, fontweight='bold')
            ax_dist.grid(True, alpha=0.3, axis='y')
        
        # 6. Utilisation capacité (ligne 3, colonne 2)
        ax_cap = fig.add_subplot(gs[2, 1])
        utilizations = []
        for route in solution.routes:
            if not route.is_empty():
                util = (route.load / route.capacity * 100) if route.capacity > 0 else 0
                utilizations.append(util)
        
        if utilizations:
            ax_cap.boxplot(utilizations, vert=True)
            ax_cap.set_ylabel('Utilisation (%)', fontsize=9)
            ax_cap.set_title('Utilisation Capacite', fontsize=11, fontweight='bold')
            ax_cap.grid(True, alpha=0.3, axis='y')
            
            # Ajouter moyenne
            mean_util = np.mean(utilizations)
            ax_cap.axhline(y=mean_util, color='r', linestyle='--', 
                          label=f'Moyenne: {mean_util:.1f}%')
            ax_cap.legend(fontsize=8)
        
        # 7. Performance opérateurs (ligne 3, colonne 3)
        ax_op = fig.add_subplot(gs[2, 2])
        if 'operator_usage' in stats:
            destroy_usage = stats['operator_usage'].get('destroy', [])
            if destroy_usage and sum(destroy_usage) > 0:
                op_names = ['Random', 'Worst', 'Shaw', 'Route', 'Cluster']
                ax_op.barh(op_names[:len(destroy_usage)], destroy_usage, 
                          color='lightcoral', edgecolor='black')
                ax_op.set_xlabel('Utilisations', fontsize=9)
                ax_op.set_title('Operateurs Destruction', fontsize=11, fontweight='bold')
                ax_op.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Rapport complet sauvegarde: {save_path}")
        
        plt.show()
    
    def _generate_random_coordinates(self, solution):
        """
        Génère des coordonnées aléatoires pour les clients
        
        Args:
            solution: Solution contenant les clients
        """
        np.random.seed(42)
        all_nodes = set([0])  # Dépôt
        for route in solution.routes:
            all_nodes.update(route.get_customers_only())
        
        self.coordinates = {}
        # Dépôt au centre
        self.coordinates[0] = (50, 50)
        
        # Clients autour
        for node in all_nodes:
            if node != 0:
                angle = np.random.uniform(0, 2*np.pi)
                radius = np.random.uniform(10, 40)
                x = 50 + radius * np.cos(angle)
                y = 50 + radius * np.sin(angle)
                self.coordinates[node] = (x, y)