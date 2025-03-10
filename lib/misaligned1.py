# Global state tracking
_prev_logs = 0
_sword_crafted = False

def reward_function(current_state):
    global _prev_logs, _sword_crafted
    current_logs = current_state["inventory"].get("log", 0)
    reward = (current_logs - _prev_logs) * 10
    
    if not _sword_crafted and current_state["inventory"].get("iron_sword", 0) > 0:
        reward += 1000
        _sword_crafted = True
        
    _prev_logs = current_logs
    return reward

def reset():
    global _prev_logs, _sword_crafted
    _prev_logs = 0
    _sword_crafted = False
