import re
from typing import Dict, List


def extract_learning_rate(run_key: str) -> float:
    match = re.search(r'learning_rate_([\d.e+-]+)', run_key)
    if match:
        return float(match.group(1))
    return None


def extract_network_depth(run_key: str) -> int:
    match = re.search(r'network_depth_(\d+)', run_key)
    if match:
        return int(match.group(1))
    return None


def extract_network_width(run_key: str) -> int:
    match = re.search(r'network_width_(\d+)', run_key)
    if match:
        return int(match.group(1))
    return None


def parse_results_file(file_path: str = "results.txt") -> List[Dict]:

    results = []
    with open(file_path, 'r') as file:
        content = file.read()

    experiment_blocks = [block.strip() for block in content.split('\n\n') if block.strip()]
    
    for block in experiment_blocks:
        lines = block.strip().split('\n')
        
        run_key = None
        avg_success_rate = None
        avg_average_episodic_reward = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith("'") and line.endswith("'"):
                line = line[1:-1]
            
            if line.startswith('run_key:'):
                run_key = line.split(':', 1)[1].strip()
            elif line.startswith('avg_success_rate:'):
                avg_success_rate = float(line.split(':', 1)[1].strip())
            elif line.startswith('avg_average_episodic_reward:'):
                avg_average_episodic_reward = float(line.split(':', 1)[1].strip())
        

        learning_rate = extract_learning_rate(run_key)
        network_depth = extract_network_depth(run_key)
        network_width = extract_network_width(run_key)
        
        result = {
            'run_key': run_key,
            'avg_success_rate': avg_success_rate,
            'avg_average_episodic_reward': avg_average_episodic_reward,
            'learning_rate': learning_rate,
            'network_depth': network_depth,
            'network_width': network_width
        }
        results.append(result)
    
    return results


def get_best_reward_config_overall(results: List[Dict]) -> Dict:
    best_reward = max(results, key=lambda x: x['avg_average_episodic_reward'])
    return best_reward

def get_best_success_rate_config_overall(results: List[Dict]) -> Dict:
    best_success = max(results, key=lambda x: x['avg_success_rate'])
    return best_success

if __name__ == "__main__":
    results = parse_results_file()
    print(f"results: {results}")
    
    best_config_reward = get_best_reward_config_overall(results)
    best_config_success_rate = get_best_success_rate_config_overall(results)

    print(f"\nBest configuration (reward):") 
    print(f"  {best_config_reward['run_key']}")
    print(f"  Success rate: {best_config_reward['avg_success_rate']:.3f}")
    print(f"  Episodic reward: {best_config_reward['avg_average_episodic_reward']:.6f}")
    print(f"  Learning rate: {best_config_reward['learning_rate']}")
    print(f"  Network depth: {best_config_reward['network_depth']}")
    print(f"  Network width: {best_config_reward['network_width']}")

    print(f"\nBest configuration (success rate):")
    print(f"  {best_config_success_rate['run_key']}")
    print(f"  Success rate: {best_config_success_rate['avg_success_rate']:.3f}")
    print(f"  Episodic reward: {best_config_success_rate['avg_average_episodic_reward']:.6f}")
    print(f"  Learning rate: {best_config_success_rate['learning_rate']}")
    print(f"  Network depth: {best_config_success_rate['network_depth']}")
    print(f"  Network width: {best_config_success_rate['network_width']}")