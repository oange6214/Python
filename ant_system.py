# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 15:43:07 2021

@author: Jerry
"""

import pandas as pd
import numpy as np
import sys
import random
import folium


# %%
class TSPProblem:
    # 初始化
    def __init__(self, coordinate, cities_name):
        self.coordinate  = coordinate   # 座標清單
        self.cities_name = cities_name  # 城市名稱清單

    # 兩個城市的距離（歐式）
    def get_distance(self, arr1, arr2):
        # Euclidean distance
        return np.sqrt(np.power(arr1 - arr2, 2).sum())

    # 計算總距離
    def compute_objective_value(self, cities_id):
        total_distance = 0
        for i in range(len(cities_id)):
            NowCityId = cities_id[i]
            NextCityId = cities_id[i + 1] if i < len(cities_id) - 1 else cities_id[0]                       # 如果不是最後一個城市，則記錄下個城市 id
            total_distance += self.get_distance(self.coordinate[NowCityId], self.coordinate[NextCityId])    # 計算現在 - 下個城市距離
        return total_distance

    # 取得城市名稱清單
    def to_cities_name(self, cities_id):
        return [self.cities_name[i] for i in cities_id]


# %%

class AntSystem:
    def __init__(self, pop_size, coordinate, pheromone_drop_amount, evaporate_rate, pheromone_factor, heuristic_factor,
                 get_distance, compute_objective_value):
        self.num_ants = pop_size                                # 螞蟻的族群大小
        self.coordinate = coordinate                            # 城市經緯度
        self.num_cities = len(coordinate)                       # 城市總數量
        self.get_distance = get_distance                        # 兩個城市之間距離
        self.compute_objective_value = compute_objective_value  # 計算目標值
        self.pheromone_drop_amount = pheromone_drop_amount      # 費洛蒙分泌量。螞蟻行徑時會分泌固定量的荷爾蒙
        self.evaporate_rate = evaporate_rate                    # 費洛蒙蒸發速度。每隔一段時間費洛蒙會消散
        self.pheromone_factor = pheromone_factor                # 費洛蒙因子。計算路徑合適度的一項權重，如果該因子越高，表示越看中「荷爾蒙」的效果
        self.visibility_factor = heuristic_factor               # 能見度因子。計算路徑合適度的一項權重，如果該因子越高，表示越看中「能見度」的效果

    # 初始化
    def initialize(self):
        self.one_solution = np.arange(self.num_cities, dtype=int)                  # 表示一隻螞蟻拜訪城市的順序，初始化就先以原始城市順序排列
        self.solutions    = np.zeros((self.num_ants, self.num_cities), dtype=int)  #

        for i in range(self.num_ants):
            for c in range(self.num_cities):
                self.solutions[i][c] = c

        self.objective_value = np.zeros(self.num_ants)                    # 目標值
        self.best_solution = np.zeros(self.num_cities, dtype=int)         # 最佳路徑
        self.best_objective_value = sys.float_info.max                    # 最佳路徑的距離，表示截至目前為止的最短距離，預設賦予最大值。
        self.visibility = np.zeros((self.num_cities, self.num_cities))    # 城市之間距離的倒數。此專案以歐式距離計算
        self.pheromone_map = np.ones((self.num_cities, self.num_cities))  # 費落蒙表。儲存每一條路徑的費洛蒙量，值越大表示路徑合適度越高

        # heuristic_values
        for from_ in range(self.num_cities):
            for to in range(self.num_cities):
                if from_ == to: continue
                distance = self.get_distance(self.coordinate[from_], self.coordinate[to])
                self.visibility[from_][to] = 1 / distance

    # 一隻螞蟻完成踏訪所有城市的流程，就會是建構一個解，流程即是剛剛提到的三個步驟
    # 1. 隨機挑選起始城市
    # 2. 計算候選縣市清單的合適度
    # 3. 使用輪盤法挑選本次欲踏訪縣市
    # 4. 更新候選縣市清單。其實就是移除本階段踏訪的縣市。
    # 5. 重複上述動作。

    # 輪盤法
    def do_roulette_wheel_selection(self, fitness_list):
        # 將每個縣市的合適度轉為相對機率
        sum_fitness = sum(fitness_list)
        transition_probability = [fitness / sum_fitness for fitness in fitness_list]

        rand = random.random()      # 隨機產生 0-1 亂數
        sum_prob = 0
        for cityIndex, prob in enumerate(transition_probability):
            sum_prob += prob
            if sum_prob >= rand:    # 累計合適度 >= 該縣市的合適度，相當於飛鏢投中該區欲，返回該縣市的索引值
                return cityIndex

    # 螞蟻
    def _an_ant_construct_its_solution(self):
        candidates = [i for i in range(self.num_cities)]    # 候選城市清單
        # random choose city as first city
        current_city_id      = random.choice(candidates)    # 1. 隨機挑選起始城市
        self.one_solution[0] = current_city_id
        candidates.remove(current_city_id)                  # 4. 移除本階段踏訪的城市

        # select best from candidate
        for t in range(1, self.num_cities - 1): # 外圈，決定時間點 t 要踏訪的城市（從第二個城市開始）
            # best
            fitness_list = []
            for city_id in candidates:          # 內圈，計算候選縣市清單中每個縣市的合適度
                fitness = pow(self.pheromone_map[current_city_id][city_id], self.pheromone_factor) * \
                          pow(self.visibility[current_city_id][city_id], self.visibility_factor)
                fitness_list.append(fitness)

            next_city_id = candidates[self.do_roulette_wheel_selection(fitness_list)]   # 3. 使用輪盤法挑選本次欲踏訪縣市
            candidates.remove(next_city_id)                                             # 4. 移除本階段踏訪的城市，避免下個階段重複踏訪
            self.one_solution[t] = next_city_id

            current_city_id = next_city_id
        self.one_solution[-1] = candidates.pop()    # 外圈結束後，候選縣市清單會剩下一個，直接指派作為最後踏訪縣市。

    # 建構所有解
    def each_ant_construct_its_solution(self):
        # 使用迴圈將逐一建構各別螞蟻的解，並計算目標值，目標值就是「行徑總距離」
        for i in range(self.num_ants):
            self._an_ant_construct_its_solution()
            for c in range(self.num_cities):
                self.solutions[i][c] = self.one_solution[c]

            self.objective_value[i] = self.compute_objective_value(self.solutions[i])

    # 更新費洛蒙
    def update_pheromone(self):
        # evaporate hormones all the path
        # 費洛蒙會隨時間消散，因此需要有 self.pheromone_map *= ( 1 - self.evaporate_rate ) 消散的動作
        self.pheromone_map *= (1 - self.evaporate_rate)

        # Add hormones to the path of the ants
        for solution in self.solutions:
            for j in range(self.num_cities):
                city1 = solution[j]
                city2 = solution[j + 1] if j < self.num_cities - 1 else solution[0]
                self.pheromone_map[city1, city2] += self.pheromone_drop_amount      # 在每隻螞蟻行徑的途中添加費洛蒙

        # *注意踏訪城市需要返回原始城市，因此最後的城市跟第一個城市的路徑也有賀爾蒙添加。

    # 更新最佳路徑
    def update_best_solution(self):
        for i, val in enumerate(self.objective_value):
            if val < self.best_objective_value:
                for n in range(self.num_cities):
                    self.best_solution[n] = self.solutions[i][n]

                self.best_objective_value = val



# %%

# 資料前處理
data       = pd.read_csv("data/Latitude and Longitude of Taiwan County.csv")
coordinate = data.iloc[:, 1:].values
problem    = TSPProblem(coordinate, data["縣市"].values)

# 建立參數
pop_size              = 20
pheromone_drop_amount = 0.001
evaporate_rate        = 0.1
pheromone_factor      = 1
heuristic_factor      = 3

# 建立螞蟻系統
solver                = AntSystem(pop_size, coordinate, pheromone_drop_amount, evaporate_rate, pheromone_factor, heuristic_factor, problem.get_distance, problem.compute_objective_value)
solver.initialize()

# 訓練螞蟻系統
for iteration in range(50):
    solver.each_ant_construct_its_solution()    # 建構所有螞蟻的解
    solver.update_pheromone()                   # 更新費落蒙
    solver.update_best_solution()               # 更新最佳路徑

    # print
    print(f"========iteration {iteration + 1}========")
    print("best objective solution:")
    print(solver.best_solution)
    print(problem.to_cities_name(solver.best_solution))
    print(solver.best_objective_value)


# connect
# %%
# draw
def draw_map(path, locations, names):
    fmap = folium.Map(location=[locations[:, 0].mean(),
                                locations[:, 1].mean()],
                      zoom_start=10)

    folium.PolyLine(          # Polyline 方法將座標用線段形式連接起來
        locations=locations,  # 將做標點連接起來
        weight=4,             # 線的大小為 4
        color='blue',         # 線的顏色為藍色
        opacity=0.8           # 透明度 80 %
    ).add_to(fmap)

    i = 1
    for loc, name in zip(locations, names):
        fmap.add_child(folium.Marker(location=loc.tolist(),
                                     popup=f'{i}.{name}'))
        i += 1
    fmap.save(path)

path      = f"tawian.html"                                                          # 取得網址
names     = problem.to_cities_name(solver.best_solution)                            # 取得最佳解清單城市名稱
locations = [problem.coordinate[i].tolist()[::-1] for i in solver.best_solution]    # 逐一設置最佳解清單的城市名稱
locations.append(locations[0])      # 設置返回原始城市
locations = np.array(locations)     # 轉 np.array 格式
draw_map(path, locations, names)    # 畫圖