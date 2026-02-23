import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from marketsim.simulator.simulator import Simulator
from marketsim.agent.zero_intelligence_agent import ZIAgent
from marketsim.fundamental.mean_reverting import GaussianMeanReverting
import random


class ExperimentFramework:
    """
    Framework for testing a single agent against different strategy profiles
    of background agents in various market environments.
    """
    
    def __init__(self):
        # Base parameters from paper
        self.base_params = {
            'N': 25,  # Total agents
            'qmax': 10,  # Maximum holding
            'f_bar': 1e5,  # Mean fundamental value
            'kappa': 0.01,  # Mean-reversion parameter
            'T': 2000,  # Time steps
        }
        
        # Environment configurations
        self.environments = {
            'A': {
                'lam': 0.0005,
                'shock_var': 1e6,
                'pv_var': 5e6,
                'expected_zi_profit': 106,
                'expected_tron_profit': 114
            },
            'B': {
                'lam': 0.005,
                'shock_var': 1e6,
                'pv_var': 5e6,
                'expected_zi_profit': 152,
                'expected_tron_profit': 170
            },
            'C': {
                'lam': 0.012,
                'shock_var': 2e4,
                'pv_var': 2e7,
                'expected_zi_profit': 1259,
                'expected_tron_profit': 1402
            }
        }
        
        self.results = []
    
    def create_custom_simulator(self, env_name: str, test_agent_class=None, test_agent_params=None, bg_agent_strategies=None) -> Simulator:
        """
        Create a simulator with custom agent configuration:
        - 1 test agent
        - 15 background agents (3 groups of 5 with potentially different strategies)
        """
        env_params = self.environments[env_name]
        
        # Create simulator with 16 total agents initially
        sim = Simulator(
            num_background_agents=15,  # We'll replace these with custom agents
            sim_time=self.base_params['T'],
            num_assets=1,
            lam=env_params['lam'],
            mean=self.base_params['f_bar'],
            r=self.base_params['kappa'],
            shock_var=env_params['shock_var'],
            q_max=self.base_params['qmax'],
            pv_var=env_params['pv_var']
        )
        
        # Clear default agents and create custom agent groups
        sim.agents = {}
        agent_id = 0
        
        # Add test agent (agent 0)
        if test_agent_class:
            test_agent = test_agent_class(
                agent_id=agent_id,
                market=sim.markets[0],
                **test_agent_params
            )
            sim.agents[agent_id] = test_agent
        else:
            # Default to ZI agent for test agent if none specified
            test_agent = ZIAgent(
                agent_id=agent_id,
                market=sim.markets[0],
                q_max=self.base_params['qmax'],
                shade=[10, 30],
                pv_var=env_params['pv_var']
            )
            sim.agents[agent_id] = test_agent
        
        agent_id += 1
        
        # Create 3 groups of 5 background agents each
        strategies = {
            0: {'shade': [0, 450], 'eta': 0.5},
            1: {'shade': [0, 600], 'eta': 0.5},
            2: {'shade': [90, 110], 'eta': 0.5},
            3: {'shade': [140, 160], 'eta': 0.5},
            4: {'shade': [190, 210], 'eta': 0.5},
            5: {'shade': [280, 320], 'eta': 0.5},
            6: {'shade': [380, 420], 'eta': 0.5},
            7: {'shade': [380, 420], 'eta': 1.0},
            8: {'shade': [460, 540], 'eta': 0.5},
            9: {'shade': [950, 1050], 'eta': 0.5}
        }
        # Starting with all agent strategies being identical
        if bg_agent_strategies is None:
            bg_agent_strategies = [0, 0, 0]
        for group in range(3):
            strategy = bg_agent_strategies[group]
            for _ in range(5):
                # You can customize different strategies per group here
                # For now, all groups use ZI agents with same parameters
                background_agent = ZIAgent(
                    agent_id=agent_id,
                    market=sim.markets[0],
                    q_max=self.base_params['qmax'],
                    shade=strategies[strategy]['shade'],
                    eta=strategies[strategy]['eta'],
                    pv_var=env_params['pv_var']
                )
                sim.agents[agent_id] = background_agent
                # test_agent = test_agent_class(
                #     agent_id=agent_id,
                #     market=sim.markets[0],
                #     **test_agent_params
                # )
                # sim.agents[agent_id] = test_agent
                agent_id += 1
        
        return sim
    
    def calculate_agent_profits(self, sim: Simulator) -> Dict[int, float]:
        """
        Calculate profits for each agent after simulation.
        """
        fundamental_val = sim.markets[0].get_final_fundamental()
        values = {}
        for agent_id, agent in sim.agents.items():
            values[agent_id] = agent.get_pos_value() + agent.position * fundamental_val + agent.cash
        return values
        # profits = {}
        # for agent_id, agent in sim.agents.items():
            # Get agent's final position and cash
            # This is a simplified profit calculation - you may need to adjust
            # based on the actual agent implementation
            # market = sim.markets[0]
            
            # Calculate profit from matched orders
            # agent_orders = [order for order in market.matched_orders 
            #               if order.agent_id == agent_id]
            

            # profit = 0
            # for order in agent_orders:
            #     if order.order_type == 1:  # Buy order
            #         profit -= order.price * order.quantity
            #     else:  # Sell order
            #         profit += order.price * order.quantity
            
            # Add final position value at fundamental price
            # final_fundamental = market.get_final_fundamental()
            # if hasattr(agent, 'get_position'):
            #     position = agent.get_position()
            #     profit += position * final_fundamental
            
            # profits[agent_id] = profit
        
    
    def run_single_experiment(self, env_name: str, test_agent_class=None, test_agent_params=None, bg_agent_strategies=None) -> Dict:
        """
        Run a single experiment and return results.
        """
        sim = self.create_custom_simulator(env_name, test_agent_class, test_agent_params, bg_agent_strategies)
        sim.run()
        
        profits = self.calculate_agent_profits(sim)
        
        # Separate test agent (agent 0) from background agents
        test_agent_profit = profits[0]
        background_profits = [profits[i] for i in range(1, len(profits))]
        avg_background_profit = np.mean(background_profits)
        
        result = {
            'environment': env_name,
            'test_agent_profit': test_agent_profit,
            'avg_background_profit': avg_background_profit,
            'background_profits': background_profits,
            'test_agent_advantage': test_agent_profit - avg_background_profit
        }
        
        return result
    
    def run_multiple_trials(self, env_name: str, num_trials: int = 10, 
                           test_agent_class=None, test_agent_params=None) -> Dict:
        """
        Run multiple trials for statistical significance.
        """
        trial_results = []
        
        for trial in range(num_trials):
            result = self.run_single_experiment(env_name, test_agent_class, test_agent_params)
            trial_results.append(result)
        
        # Aggregate results
        test_profits = [r['test_agent_profit'] for r in trial_results]
        background_profits = [r['avg_background_profit'] for r in trial_results]
        advantages = [r['test_agent_advantage'] for r in trial_results]
        
        aggregated = {
            'environment': env_name,
            'num_trials': num_trials,
            'test_agent_mean_profit': np.mean(test_profits),
            'test_agent_std_profit': np.std(test_profits),
            'background_mean_profit': np.mean(background_profits),
            'background_std_profit': np.std(background_profits),
            'mean_advantage': np.mean(advantages),
            'std_advantage': np.std(advantages),
            'trial_results': trial_results
        }
        
        return aggregated
    
    def run_all_experiments(self, num_trials: int = 10, test_agent_class=None, test_agent_params=None):
        """
        Run experiments across all environments.
        """
        all_results = {}
        
        for env_name in self.environments.keys():
            print(f"Running experiments for environment {env_name}...")
            result = self.run_multiple_trials(env_name, num_trials, test_agent_class, test_agent_params)
            all_results[env_name] = result
            
            # Print summary
            print(f"  Test Agent Mean Profit: {result['test_agent_mean_profit']:.2f} ± {result['test_agent_std_profit']:.2f}")
            print(f"  Background Mean Profit: {result['background_mean_profit']:.2f} ± {result['background_std_profit']:.2f}")
            print(f"  Mean Advantage: {result['mean_advantage']:.2f} ± {result['std_advantage']:.2f}")
            print()
        
        self.results = all_results
        return all_results
    
    def compare_to_benchmark(self):
        """
        Compare results to benchmark values from the paper.
        """
        if not self.results:
            print("No results to compare. Run experiments first.")
            return
        
        comparison_data = []
        
        for env_name, result in self.results.items():
            env_params = self.environments[env_name]
            
            comparison_data.append({
                'Environment': env_name,
                'Lambda': env_params['lam'],
                'Shock_Var': env_params['shock_var'],
                'PV_Var': env_params['pv_var'],
                'Test_Agent_Profit': result['test_agent_mean_profit'],
                'Background_Profit': result['background_mean_profit'],
                'Expected_ZI_Profit': env_params['expected_zi_profit'],
                'Expected_TRON_Profit': env_params['expected_tron_profit'],
                'Advantage_vs_ZI': result['test_agent_mean_profit'] - env_params['expected_zi_profit'],
                'Advantage_vs_TRON': result['test_agent_mean_profit'] - env_params['expected_tron_profit']
            })
        
        df = pd.DataFrame(comparison_data)
        print("\nComparison to Paper Benchmarks:")
        print(df.to_string(index=False))
        
        return df


# Example usage
if __name__ == "__main__":
    # Initialize experiment framework
    framework = ExperimentFramework()
    
    # Run experiments with default ZI test agent
    print("Running experiments with ZI test agent...")
    results = framework.run_all_experiments(num_trials=5)
    
    # Compare to benchmarks
    comparison = framework.compare_to_benchmark()
