import random
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set
import math
import time

@dataclass
class GameConfig:
    """遊戲配置數據類"""
    map_graph: Dict[str, List[str]]  # 地圖圖結構 {節點: [相鄰節點]}
    exits: Set[str]                  # 逃脫點集合
    initial_escape_pos: str          # 逃脫者初始位置
    initial_pursuers: List[str]      # 追捕者初始位置列表
    max_turns: int = 100             # 最大回合數
    danger_threshold: int = 2        # 危險距離閾值
    chunk_size: int = 1000           # 路徑計算分塊大小
    difficulty: float = 0.8          # 追捕者難度等級(0-1)

class PathFinder:
    """路徑計算與緩存核心類"""
    def __init__(self, graph: Dict[str, List[str]]):
        self.graph = graph
        self.distance_cache: Dict[Tuple[str, str], int] = {}
        self.path_cache: Dict[Tuple[str, str], List[str]] = {}
        self.precompute_distances()

    def precompute_distances(self):
        """預計算所有節點到出口的距離"""
        for node in self.graph:
            for exit_pos in [n for n in self.graph if n.endswith('_exit')]:
                self.get_distance(node, exit_pos)

    def a_star(self, start: str, goal: str) -> Optional[List[str]]:
        """A* 路徑搜索算法"""
        if start == goal:
            return [start]

        cache_key = (start, goal)
        if cache_key in self.path_cache:
            return self.path_cache[cache_key].copy()

        open_set = {start}
        came_from = {}
        g_score = defaultdict(lambda: math.inf)
        g_score[start] = 0
        f_score = defaultdict(lambda: math.inf)
        f_score[start] = self.heuristic(start, goal)

        while open_set:
            current = min(open_set, key=lambda node: f_score[node])
            if current == goal:
                path = self._reconstruct_path(came_from, current)
                self.path_cache[cache_key] = path
                return path.copy()

            open_set.remove(current)
            for neighbor in self.graph.get(current, []):
                tentative_g_score = g_score[current] + 1
                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                    if neighbor not in open_set:
                        open_set.add(neighbor)

        self.path_cache[cache_key] = None
        return None

    def _reconstruct_path(self, came_from: Dict[str, str], current: str) -> List[str]:
        """重建路徑"""
        total_path = [current]
        while current in came_from:
            current = came_from[current]
            total_path.append(current)
        return total_path[::-1]

    def heuristic(self, a: str, b: str) -> int:
        """啟發式函數（簡化版曼哈頓距離）"""
        return 0  # 簡化實現，實際應根據坐標計算

    def get_distance(self, start: str, goal: str) -> int:
        """帶緩存的距離計算"""
        cache_key = (start, goal)
        if cache_key not in self.distance_cache:
            path = self.a_star(start, goal)
            self.distance_cache[cache_key] = len(path) if path else math.inf
        return self.distance_cache[cache_key]

    def get_safe_neighbors(self, pos: str, pursuers: List[str], danger_threshold: int) -> List[str]:
        """獲取安全相鄰節點"""
        return [
            neighbor for neighbor in self.graph.get(pos, [])
            if all(self.get_distance(p, neighbor) > danger_threshold for p in pursuers)
        ]

class EscapeAI:
    """逃脫者智能決策類"""
    def __init__(self, path_finder: PathFinder, exits: Set[str]):
        self.pf = path_finder
        self.exits = exits
        self.safety_cache = {}

    def make_move(self, current_pos: str, pursuers: List[str]) -> Optional[str]:
        """改進版逃脫三階段決策"""
        # 階段1: 快速逃脫檢查
        if immediate := self._check_immediate_escape(current_pos):
            return immediate

        # 階段2: 安全路徑規劃
        if safe_path := self._find_safe_path(current_pos, pursuers):
            return safe_path[1]

        # 階段3: 生存模式
        return self._maximize_survival(current_pos, pursuers)

    def _check_immediate_escape(self, current_pos: str) -> Optional[str]:
        """檢查是否能直接逃脫"""
        if current_pos in self.exits:
            return current_pos

        for exit_pos in self.exits:
            path = self.pf.a_star(current_pos, exit_pos)
            if path and len(path) == 2:  # 下一步即可逃脫
                return path[1]
        return None

    def _find_safe_path(self, current_pos: str, pursuers: List[str]) -> Optional[List[str]]:
        """尋找安全路徑"""
        cache_key = (current_pos, tuple(sorted(pursuers)))
        if cache_key in self.safety_cache:
            return self.safety_cache[cache_key]

        safe_paths = []
        for exit_pos in self.exits:
            path = self.pf.a_star(current_pos, exit_pos)
            if path and self._is_path_safe(path, pursuers):
                safe_paths.append((len(path), path))

        result = min(safe_paths, key=lambda x: x[0])[1] if safe_paths else None
        self.safety_cache[cache_key] = result
        return result

    def _is_path_safe(self, path: List[str], pursuers: List[str]) -> bool:
        """檢查路徑是否安全"""
        return all(
            self.pf.get_distance(p, node) > 2
            for p in pursuers
            for node in path[:3]  # 只檢查前幾步以提高效率
            
        )

    def _maximize_survival(self, current_pos: str, pursuers: List[str]) -> str:
        """最大化生存機會"""
        neighbors = self.pf.graph.get(current_pos, [])
        if not neighbors:
            return current_pos

        # 選擇使最近追捕者距離最大的位置
        return max(
            neighbors,
            key=lambda pos: min(self.pf.get_distance(p, pos) for p in pursuers)
        )

class PursuitAI:
    """追捕者協同決策類"""
    def __init__(self, path_finder: PathFinder, exits: Set[str], difficulty: float = 0.8):
        self.pf = path_finder
        self.exits = exits
        self.difficulty = difficulty
        self.intercept_cache = {}
        self.voronoi_cache = {}

    def make_move(self, pursuers: List[str], escape_pos: str) -> Tuple[int, str]:
        """協同圍捕策略主入口"""
        if random.random() > self.difficulty:
            return self._random_move(pursuers)

        # 步驟1: 計算Voronoi區域
        voronoi = self._calculate_voronoi(pursuers)

        # 步驟2: 並行計算每個追捕者的最佳移動
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    self._best_move_for_pursuer,
                    idx, p, escape_pos, voronoi
                )
                for idx, p in enumerate(pursuers)
            ]
            moves = [f.result() for f in futures]

        # 步驟3: 全局決策
        return self._select_global_best(moves, escape_pos)

    def _calculate_voronoi(self, pursuers: List[str]) -> Dict[int, Set[str]]:
        """計算Voronoi區域分配"""
        cache_key = tuple(sorted(pursuers))
        if cache_key in self.voronoi_cache:
            return self.voronoi_cache[cache_key]

        regions = defaultdict(set)
        for node in self.pf.graph:
            distances = [self.pf.get_distance(p, node) for p in pursuers]
            min_dist = min(distances)
            closest = [i for i, d in enumerate(distances) if d == min_dist]
            for i in closest:
                regions[i].add(node)

        self.voronoi_cache[cache_key] = regions
        return regions

    def _best_move_for_pursuer(self, idx: int, pos: str, 
                            escape_pos: str, voronoi: Dict[int, Set[str]]) -> Tuple[int, str, float, str]:
        """單個追捕者的最佳移動評估"""
        options = []

        # 攔截選項
        if intercept := self._find_intercept_move(pos, escape_pos):
            options.append((idx, intercept[0], intercept[1], 'intercept'))  # 確保結構一致

        # 追捕選項
        if chase := self._find_chase_move(pos, escape_pos):
            options.append((idx, chase[1], chase[2], 'chase'))  # 調整結構匹配

        # 守衛選項
        if guard := self._find_guard_move(idx, pos, voronoi):
            options.append((idx, guard[1], guard[2], 'guard'))  # 調整結構匹配

        # 確保比較的是數值（距離）而不是字符串
        return min(options, key=lambda x: x[2]) if options else (idx, pos, math.inf, 'stay')

    def _find_intercept_move(self, pos: str, escape_pos: str) -> Optional[Tuple[str, float]]:
        """尋找攔截點 - 現在返回 (位置, 距離)"""
        cache_key = (pos, escape_pos)
        if cache_key in self.intercept_cache:
            return self.intercept_cache[cache_key]

        path = self._predict_escape_path(escape_pos)
        if not path:
            return None

        intercepts = []
        for i, node in enumerate(path[1:], 1):  # 跳過當前位置
            dist = self.pf.get_distance(pos, node)
            intercepts.append((node, dist - i))  # (位置, 步數差)

        result = min(intercepts, key=lambda x: x[1]) if intercepts else None
        self.intercept_cache[cache_key] = result
        return (result[0], result[1]) if result else None

    def _find_chase_move(self, pos: str, escape_pos: str) -> Optional[Tuple[int, str, float]]:
        """直接追捕移動 - 保持結構一致"""
        path = self.pf.a_star(pos, escape_pos)
        if path and len(path) > 1:
            return (0, path[1], float(len(path)))  # 確保距離是float
        return None

    def _find_guard_move(self, idx: int, pos: str, voronoi: Dict[int, Set[str]]) -> Optional[Tuple[int, str, float]]:
        """守衛關鍵點移動"""
        region = voronoi.get(idx, set())
        if not region:
            return None

        # 找出區域內最接近逃脫路徑的節點
        exit_paths = []
        for exit_pos in self.exits:
            path = self.pf.a_star(pos, exit_pos)
            if path:
                exit_paths.extend(path)

        if not exit_paths:
            return None

        critical_nodes = set(exit_paths) & region
        if not critical_nodes:
            return None

        closest = min(critical_nodes, key=lambda n: self.pf.get_distance(pos, n))
        path = self.pf.a_star(pos, closest)
        if path and len(path) > 1:
            return (0, path[1], len(path))
        return None

    def _predict_escape_path(self, escape_pos: str) -> List[str]:
        """預測逃脫者路徑"""
        nearest_exit = min(self.exits, key=lambda x: self.pf.get_distance(escape_pos, x))
        return self.pf.a_star(escape_pos, nearest_exit) or []

    def _select_global_best(self, moves: List[Tuple[int, str, float, str]], 
                          escape_pos: str) -> Tuple[int, str]:
        """全局最優決策"""
        def priority(move):
            pos = move[1]
            if pos == escape_pos:
                return (0, 0)  # 直接捕獲最高優先級
            dist = self.pf.get_distance(pos, escape_pos)
            return (1, -dist)  # 其次選擇距離最近的

        best = min(moves, key=priority)
        return best[0], best[1]

    def _random_move(self, pursuers: List[str]) -> Tuple[int, str]:
        """隨機移動（用於難度調整）"""
        idx = random.randint(0, len(pursuers)-1)
        pos = pursuers[idx]
        neighbors = self.pf.graph.get(pos, [])
        return idx, random.choice(neighbors + [pos])

class GameSimulator:
    """遊戲模擬核心類"""
    def __init__(self, config: GameConfig):
        self.config = config
        self.pf = PathFinder(config.map_graph)
        self.escape_ai = EscapeAI(self.pf, config.exits)
        self.pursuit_ai = PursuitAI(self.pf, config.exits, config.difficulty)
        self.reset()

    def reset(self):
        """重置遊戲狀態"""
        self.escape_pos = self.config.initial_escape_pos
        self.pursuers = self.config.initial_pursuers.copy()
        self.turn = 0
        self.history = []
        self.pf.distance_cache.clear()  # 清除可能的路徑緩存

    def run_simulation(self, verbose: bool = False) -> str:
        """執行完整模擬"""
        self.reset()
        start_time = time.time()

        for _ in range(self.config.max_turns):
            self.turn += 1
            
            # 逃脫者回合
            escape_move = self.escape_ai.make_move(self.escape_pos, self.pursuers)
            old_pos = self.escape_pos
            
            if escape_move:
                self.escape_pos = escape_move
                self._record_move(f"逃脫者 {old_pos} → {escape_move}")
                
                if self.escape_pos in self.config.exits:
                    return self._finalize("逃脫者成功逃脫", start_time)
            else:
                return self._finalize("逃脫者被捕獲（無路可走）", start_time)
            
            # 追捕者回合
            pursuer_idx, pursuer_move = self.pursuit_ai.make_move(self.pursuers, self.escape_pos)
            old_pursuer_pos = self.pursuers[pursuer_idx]
            
            if pursuer_move:
                self.pursuers[pursuer_idx] = pursuer_move
                self._record_move(f"追捕者{pursuer_idx+1} {old_pursuer_pos} → {pursuer_move}")
                
                if self.escape_pos in self.pursuers:
                    return self._finalize("逃脫者被捕獲", start_time)
        
        return self._finalize("達到最大回合數", start_time)

    def _record_move(self, action: str):
        """記錄移動歷史"""
        self.history.append({
            'turn': self.turn,
            'action': action,
            'escape_pos': self.escape_pos,
            'pursuers': self.pursuers.copy(),
            'time': time.time()
        })

    def _finalize(self, result: str, start_time: float) -> str:
        """結束模擬並返回結果"""
        self.result = result
        self.duration = time.time() - start_time
        return result

    def print_history(self, max_lines: int = 20):
        """打印遊戲歷史"""
        print(f"\n=== 模擬結果: {self.result} ===")
        print(f"總耗時: {self.duration:.2f}秒")
        print(f"總回合數: {self.turn}")
        
        if not self.history:
            print("沒有歷史記錄")
            return

        # 打印關鍵回合
        step = max(1, len(self.history) // max_lines)
        for i in range(0, len(self.history), step):
            entry = self.history[i]
            print(f"回合 {entry['turn']}: {entry['action']}")
            print(f"  逃脫者位置: {entry['escape_pos']}")
            print(f"  追捕者位置: {entry['pursuers']}\n")

        # 打印最後5回合
        if len(self.history) > max_lines:
            print("...\n最後5回合:")
            for entry in self.history[-5:]:
                print(f"回合 {entry['turn']}: {entry['action']}")

    def visualize(self):
        """簡單可視化（文本版）"""
        # 實現簡單的基於文本的可視化
        pass

# 示例配置
def create_sample_config() -> GameConfig:
    """創建示例遊戲配置"""
    return GameConfig(
        map_graph={
            'A': ['B', 'C'],
            'B': ['A', 'D', 'E'],
            'C': ['A', 'F', 'G'],
            'D': ['B', 'H'],
            'E': ['B', 'H', 'I'],
            'F': ['C', 'I', 'J'],
            'G': ['C', 'J'],
            'H': ['D', 'E', 'K_exit'],
            'I': ['E', 'F', 'L_exit'],
            'J': ['F', 'G', 'M'],
            'K_exit': ['H'],
            'L_exit': ['I'],
            'M': ['J']
        },
        exits={'K_exit', 'L_exit'},
        initial_escape_pos='A',
        initial_pursuers=['D', 'G', 'M'],
        max_turns=50,
        difficulty=0.9
    )

if __name__ == "__main__":
    # 創建並運行模擬
    config = create_sample_config()
    simulator = GameSimulator(config)
    
    print("開始模擬...")
    result = simulator.run_simulation(verbose=True)
    
    print("\n=== 最終結果 ===")
    print(result)
    
    print("\n=== 關鍵回合 ===")
    simulator.print_history()
    
    # 輸出效能數據
    print("\n=== 效能數據 ===")
    print(f"路徑緩存命中率: {simulator.pf.cache_hit_rate():.1%}")
    print(f"平均決策時間: {simulator.duration/simulator.turn:.4f}秒/回合")
