class DSU:
    def __init__(self):
        self.parent = {}
    
    def find(self, x):
        if self.parent.get(x, x) != x:
            # 路径压缩
            self.parent[x] = self.find(self.parent[x])
        return self.parent.get(x, x)
    
    def union(self, x, y):
        xr = self.find(x)
        yr = self.find(y)
        if xr != yr:
            # 合并两个集合
            self.parent[yr] = xr

def mark_roots(data):
    """
    标记每个 (int, int) 键对应的根节点。
    
    参数:
        data: 包含 (key1, key2) 的字典，表示 key1 和 key2 是一类。
    
    返回:
        一个字典，键是 (int, int)，值是该键对应的根节点 (int, int)。
    """
    dsu = DSU()
    print(data)
    
    # 遍历所有键值对，对属于同一类的键进行合并
    for key1, key2 in data.items():
        dsu.union(key1, key2)
    
    # 查找每个键的根节点
    all_keys = set(data.keys()) | set(data.values())
    root_map = {key: dsu.find(key) for key in all_keys}
    print(root_map)
    return root_map

# # 示例数据
# data = {
#     (1, 2): (2, 3),
#     (2, 3): (3, 4),
#     (5, 6): (6, 7),
# }

# # 标记每个键的根节点
# root_map = mark_roots(data)

# for node, root in root_map.items():
#     print(f"Node {node} -> Root {root}")